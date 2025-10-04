import asyncio
import base64
import itertools
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, AsyncGenerator, Union

import aiofiles
import httpx
import uvicorn
import yaml
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from loguru import logger

# ===========================
# Constants
# ===========================

class Constants:
    TOKEN_BUFFER_MS = 60 * 1000
    MAX_RETRIES = 3
    BASE_RETRY_DELAY = 1.0
    REQUEST_TIMEOUT = 300.0
    DEFAULT_TIMEOUT = 30.0
    MAX_KEEPALIVE_CONNECTIONS = 10
    MAX_CONNECTIONS = 50
    IFLOW_CLIENT_ID = "10009311001"
    IFLOW_CLIENT_SECRET = "4Z3YjXycVsQvyGF1etiNlIBB4RsqSDtW"
    IFLOW_TOKEN_ENDPOINT = "https://iflow.cn/oauth/token"
    IFLOW_BASE_URL = "https://apis.iflow.cn/v1"
    IFLOW_MAX_TOKENS = 65536
    IFLOW_TEMPERATURE = 0.05
    ERROR_RESPONSE_MAX_LENGTH = 500
    ERROR_TEXT_MAX_LENGTH = 1000
    RATE_LIMIT_STATUS_CODES = {429, 500}


# ===========================
# Configuration Management
# ===========================

class Settings:
    @staticmethod
    def load_config() -> Dict[str, Any]:
        try:
            with open("config.yaml", "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.error("config.yaml not found!")
            return {}
        except Exception as e:
            logger.error(f"Error loading config.yaml: {e}")
            return {}

    config = load_config()
    API_KEY: Optional[str] = config.get("api_key")
    IFLOW_ACCOUNTS: List[str] = config.get("iflow", {}).get("accounts", [])
    IFLOW_DEFAULT_MODEL: str = config.get("iflow", {}).get("default_model", "qwen3-coder-plus")
    IFLOW_AVAILABLE_MODELS: List[str] = config.get("iflow", {}).get("available_models", ["qwen3-coder-plus"])
    IFLOW_INITIAL_BACKOFF: int = config.get("iflow", {}).get("initial_backoff_seconds", 60)
    ACCOUNT_STATE_FILE: str = "iflow_account_states.json"


logger.remove()
logger.add(sys.stderr, level="DEBUG")


# ===========================
# Utility Functions
# ===========================

async def read_json_file(path: str) -> Dict[str, Any]:
    async with aiofiles.open(path, 'r') as f:
        content = await f.read()
        return json.loads(content)


async def write_json_file(path: str, data: Dict[str, Any]) -> None:
    async with aiofiles.open(path, 'w') as f:
        await f.write(json.dumps(data, indent=2))


def calculate_exponential_backoff(failure_count: int, initial_backoff: int) -> float:
    return initial_backoff * (2 ** (failure_count - 1))


def is_rate_limit_error(status_code: int) -> bool:
    return status_code in Constants.RATE_LIMIT_STATUS_CODES


# ===========================
# Credential Management
# ===========================

@dataclass
class OAuthCredentials:
    access_token: str
    refresh_token: str
    token_type: str
    expiry_date: int
    scope: Optional[str] = None
    resource_url: Optional[str] = None
    apiKey: Optional[str] = None


class TokenValidator:
    @staticmethod
    def is_token_valid(credentials: OAuthCredentials) -> bool:
        if not credentials.expiry_date:
            return False
        return time.time() * 1000 < credentials.expiry_date - Constants.TOKEN_BUFFER_MS


class IFlowTokenRefresher:
    def __init__(self, http_client: httpx.AsyncClient):
        self.http_client = http_client

    async def refresh_token(self, credentials: OAuthCredentials, path: str) -> OAuthCredentials:
        logger.info(f"Refreshing iFlow token for {path}")
        if not credentials.refresh_token:
            raise HTTPException(status_code=500, detail=f"No refresh token in {path}")

        body_data = self._build_refresh_request(credentials.refresh_token)
        headers = self._build_request_headers()

        for attempt in range(Constants.MAX_RETRIES):
            try:
                if attempt > 0:
                    await self._apply_retry_delay(attempt)
                response = await self._execute_refresh_request(headers, body_data)
                if self._is_captcha_response(response):
                    if not self._should_retry(attempt):
                        raise HTTPException(status_code=503, detail="Token refresh blocked by anti-bot protection")
                    continue
                if not response.is_success:
                    if not self._should_retry(attempt):
                        raise HTTPException(status_code=500, detail=f"Token refresh failed: HTTP {response.status_code}")
                    self._log_http_error(response)
                    continue
                token_data = self._parse_token_response(response, attempt)
                if token_data is None:
                    continue
                if self._has_oauth_error(token_data, attempt):
                    continue
                new_credentials = self._create_new_credentials(token_data, credentials)
                await self._save_credentials(new_credentials, path)
                return new_credentials
            except httpx.RequestError as e:
                if not self._should_retry(attempt):
                    raise HTTPException(status_code=503, detail=f"Network error: {e}")
                logger.error(f"Network error during token refresh (attempt {attempt + 1}): {e}")
            except HTTPException:
                raise
            except Exception as e:
                if not self._should_retry(attempt):
                    raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
                logger.exception(f"Unexpected error (attempt {attempt + 1}): {e}")
        raise HTTPException(status_code=500, detail="Token refresh failed after all retries")

    @staticmethod
    def _build_refresh_request(refresh_token: str) -> Dict[str, str]:
        return {"grant_type": "refresh_token", "refresh_token": refresh_token}

    @staticmethod
    def _build_request_headers() -> Dict[str, str]:
        auth_string = f"{Constants.IFLOW_CLIENT_ID}:{Constants.IFLOW_CLIENT_SECRET}"
        auth_b64 = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')
        return {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Authorization': f'Basic {auth_b64}',
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }

    async def _execute_refresh_request(self, headers: Dict[str, str], body_data: Dict[str, str]) -> httpx.Response:
        return await self.http_client.post(Constants.IFLOW_TOKEN_ENDPOINT, headers=headers, data=body_data)

    @staticmethod
    def _is_captcha_response(response: httpx.Response) -> bool:
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' in content_type or 'text/plain' in content_type:
            response_text = response.text[:Constants.ERROR_RESPONSE_MAX_LENGTH]
            logger.error(f"Received HTML/text response (likely CAPTCHA): {response_text}")
            return True
        return False

    @staticmethod
    def _log_http_error(response: httpx.Response) -> None:
        error_text = response.text[:Constants.ERROR_TEXT_MAX_LENGTH]
        logger.error(f"Token refresh HTTP error {response.status_code}: {error_text}")

    @staticmethod
    def _parse_token_response(response: httpx.Response, attempt: int) -> Optional[Dict[str, Any]]:
        try:
            return response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            if not IFlowTokenRefresher._should_retry(attempt):
                raise HTTPException(status_code=500, detail="Invalid JSON response from token refresh endpoint")
            return None

    @staticmethod
    def _has_oauth_error(token_data: Dict[str, Any], attempt: int) -> bool:
        if 'error' in token_data:
            error_msg = f"{token_data.get('error')}: {token_data.get('error_description', 'No description')}"
            logger.error(f"OAuth error: {error_msg}")
            if not IFlowTokenRefresher._should_retry(attempt):
                raise HTTPException(status_code=400, detail=f"OAuth error: {error_msg}")
            return True
        return False

    @staticmethod
    def _create_new_credentials(token_data: Dict[str, Any], old_credentials: OAuthCredentials) -> OAuthCredentials:
        return OAuthCredentials(
            access_token=token_data["access_token"],
            token_type=token_data["token_type"],
            refresh_token=token_data.get("refresh_token", old_credentials.refresh_token),
            expiry_date=int(time.time() * 1000) + token_data["expires_in"] * 1000,
            scope=token_data.get("scope", old_credentials.scope),
            resource_url=old_credentials.resource_url,
            apiKey=token_data.get("apiKey", old_credentials.apiKey)
        )

    @staticmethod
    async def _save_credentials(credentials: OAuthCredentials, path: str) -> None:
        try:
            await write_json_file(path, asdict(credentials))
            logger.info(f"Token for {path} refreshed and saved.")
        except Exception as save_error:
            logger.warning(f"Failed to save refreshed credentials: {save_error}")

    @staticmethod
    async def _apply_retry_delay(attempt: int) -> None:
        delay = Constants.BASE_RETRY_DELAY * (2 ** (attempt - 1)) + (time.time() % 1)
        logger.info(f"Retry attempt {attempt + 1}/{Constants.MAX_RETRIES} after {delay:.2f}s delay")
        await asyncio.sleep(delay)

    @staticmethod
    def _should_retry(attempt: int) -> bool:
        return attempt < Constants.MAX_RETRIES - 1


class CredentialManager:
    def __init__(self) -> None:
        self._credentials_cache: Dict[str, OAuthCredentials] = {}
        self._refresh_locks: Dict[str, asyncio.Lock] = {}
        self._refresh_promises: Dict[str, asyncio.Task] = {}
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(Constants.DEFAULT_TIMEOUT),
            limits=httpx.Limits(max_keepalive_connections=Constants.MAX_KEEPALIVE_CONNECTIONS, max_connections=Constants.MAX_CONNECTIONS)
        )
        self.validator = TokenValidator()
        self.refresher = IFlowTokenRefresher(self.http_client)

    async def close(self):
        for promise in self._refresh_promises.values():
            if not promise.done():
                promise.cancel()
        self._refresh_promises.clear()
        await self.http_client.aclose()

    async def get_credentials(self, creds_path: str) -> OAuthCredentials:
        expanded_path = os.path.expanduser(creds_path)
        if expanded_path not in self._refresh_locks:
            self._refresh_locks[expanded_path] = asyncio.Lock()
        async with self._refresh_locks[expanded_path]:
            if self._has_valid_cached_credentials(expanded_path):
                return self._credentials_cache[expanded_path]
            credentials = await self._load_credentials_from_file(expanded_path)
            if not self.validator.is_token_valid(credentials):
                logger.info(f"Token for {expanded_path} is expired, refreshing.")
                credentials = await self._refresh_with_deduplication(credentials, expanded_path)
            self._credentials_cache[expanded_path] = credentials
            return credentials

    def _has_valid_cached_credentials(self, path: str) -> bool:
        return path in self._credentials_cache and self.validator.is_token_valid(self._credentials_cache[path])

    @staticmethod
    async def _load_credentials_from_file(path: str) -> OAuthCredentials:
        try:
            file_data = await read_json_file(path)
            return OAuthCredentials(**file_data)
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            raise HTTPException(status_code=500, detail=f"Failed to load credentials from {path}: {e}")

    async def _refresh_with_deduplication(self, credentials: OAuthCredentials, path: str) -> OAuthCredentials:
        if path in self._refresh_promises:
            existing_promise = self._refresh_promises[path]
            if not existing_promise.done():
                logger.info(f"Refresh already in progress for {path}, waiting...")
                try:
                    return await existing_promise
                except Exception:
                    pass
        refresh_task = asyncio.create_task(self.refresher.refresh_token(credentials, path))
        self._refresh_promises[path] = refresh_task
        try:
            return await refresh_task
        finally:
            if path in self._refresh_promises and self._refresh_promises[path] == refresh_task:
                del self._refresh_promises[path]


# ===========================
# Account Management
# ===========================

class AccountState(Enum):
    AVAILABLE = "available"
    BLOCKED = "blocked"


class Account:
    def __init__(self, creds_path: str, project_name: Optional[str] = None):
        self.creds_path = creds_path
        self.project_name = project_name
        self.status: AccountState = AccountState.AVAILABLE
        self.failure_count: int = 0
        self.available_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = {"creds_path": self.creds_path, "status": self.status.value, "failure_count": self.failure_count, "available_at": self.available_at}
        if self.project_name:
            data["project_name"] = self.project_name
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Account":
        account = cls(data["creds_path"], data.get("project_name"))
        account.status = AccountState(data.get("status", "available"))
        account.failure_count = data.get("failure_count", 0)
        account.available_at = data.get("available_at", 0.0)
        return account


class AccountManager:
    def __init__(self, accounts_config: Union[Dict[str, str], List[str]], state_file: str, initial_backoff: int):
        self.accounts: Dict[str, Account] = {}
        self.state_file = state_file
        self.initial_backoff = initial_backoff
        self._lock = asyncio.Lock()
        if isinstance(accounts_config, dict):
            self.account_cycler = itertools.cycle(accounts_config.items())
            self._init_from_dict(accounts_config)
        elif isinstance(accounts_config, list):
            self.account_cycler = itertools.cycle(accounts_config)
            self._init_from_list(accounts_config)
        self._load_states()

    def _init_from_dict(self, accounts_config: Dict[str, str]):
        for path, proj in accounts_config.items():
            self.accounts[path] = Account(path, proj)

    def _init_from_list(self, accounts_config: List[str]):
        for path in accounts_config:
            self.accounts[path] = Account(path)

    def _load_states(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    states = json.load(f)
                for path, data in states.items():
                    if path in self.accounts:
                        self.accounts[path] = Account.from_dict(data)
                logger.info(f"Loaded {len(self.accounts)} account states from {self.state_file}")
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load account states: {e}")

    async def _save_states(self):
        try:
            states = {path: acc.to_dict() for path, acc in self.accounts.items()}
            await write_json_file(self.state_file, states)
        except IOError as e:
            logger.error(f"Failed to save account states: {e}")

    async def get_available_account(self) -> Optional[Account]:
        async with self._lock:
            for _ in range(len(self.accounts)):
                creds_path = self._get_next_account_path()
                account = self.accounts[creds_path]
                if self._is_account_in_cooldown(account):
                    continue
                if account.status == AccountState.BLOCKED:
                    self._mark_account_available(account)
                return account
        return None

    def _get_next_account_path(self) -> str:
        next_item = next(self.account_cycler)
        if isinstance(next_item, tuple):
            self.account_cycler = itertools.cycle(self.accounts.keys())
            return next(self.account_cycler)
        return next_item

    @staticmethod
    def _is_account_in_cooldown(account: Account) -> bool:
        return account.status == AccountState.BLOCKED and time.time() < account.available_at

    @staticmethod
    def _mark_account_available(account: Account) -> None:
        logger.info(f"Account {account.creds_path} is now available after cooldown.")
        account.status = AccountState.AVAILABLE
        account.failure_count = 0

    async def report_failure(self, creds_path: str):
        async with self._lock:
            if creds_path in self.accounts:
                account = self.accounts[creds_path]
                account.status = AccountState.BLOCKED
                account.failure_count += 1
                backoff_duration = calculate_exponential_backoff(account.failure_count, self.initial_backoff)
                account.available_at = time.time() + backoff_duration
                logger.warning(f"Account {creds_path} hit rate limit. Failure count: {account.failure_count}. Blocking for {backoff_duration} seconds.")
                await self._save_states()


# ===========================
# API Client
# ===========================

class RequestBuilder:
    @staticmethod
    def build_request(openai_body: Dict[str, Any]) -> Dict[str, Any]:
        request_body = openai_body.copy()
        request_body["stream"] = False
        request_body["max_tokens"] = Constants.IFLOW_MAX_TOKENS
        request_body["temperature"] = Constants.IFLOW_TEMPERATURE
        return request_body


class ApiClient:
    def __init__(self, account_manager: AccountManager, cred_manager: CredentialManager):
        self.account_manager = account_manager
        self.cred_manager = cred_manager
        self.request_builder = RequestBuilder()

    async def chat_completion(self, request: Request, openai_request_body: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        num_accounts = len(self.account_manager.accounts)
        http_client: httpx.AsyncClient = request.app.state.http_client
        for attempt in range(num_accounts):
            account = await self.account_manager.get_available_account()
            if not account:
                logger.error(f"[{request_id}] No available accounts.")
                break
            logger.info(f"[{request_id}] Attempt {attempt + 1}/{num_accounts} with: {account.creds_path}")
            try:
                credentials = await self.cred_manager.get_credentials(account.creds_path)
                request_body = self.request_builder.build_request(openai_request_body)
                response = await self._execute_request(http_client, credentials, request_body, request_id)
                if response.status_code == 401:
                    response = await self._retry_with_refresh(http_client, credentials, account.creds_path, request_body, request_id)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if is_rate_limit_error(e.response.status_code):
                    await self.account_manager.report_failure(account.creds_path)
                    logger.warning(f"[{request_id}] Rate limit hit. Trying next account.")
                    continue
                else:
                    error_text = await self._read_error_response(e.response)
                    logger.exception(f"[{request_id}] HTTP error: {e.response.status_code} - {error_text}")
                    raise HTTPException(status_code=e.response.status_code, detail=f"Upstream HTTP error: {e.response.status_code}")
            except Exception as e:
                logger.exception(f"[{request_id}] Unexpected error for account.")
                raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
        logger.error(f"[{request_id}] All {num_accounts} accounts failed.")
        raise HTTPException(status_code=503, detail="All accounts are rate-limited.")

    @staticmethod
    async def _execute_request(http_client: httpx.AsyncClient, credentials: OAuthCredentials, request_body: Dict[str, Any], request_id: str) -> httpx.Response:
        """Execute HTTP request to the API."""
        # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ---
        # Используем apiKey вместо access_token для API-запросов
        if not credentials.apiKey:
            logger.error(f"[{request_id}] apiKey is missing from credentials file. Attempting to refresh.")
            # Если apiKey отсутствует, это может означать, что токен нужно обновить, чтобы его получить.
            # Вызываем исключение, чтобы запустить логику обновления.
            raise httpx.HTTPStatusError("apiKey missing", request=httpx.Request('POST', ''), response=httpx.Response(401))

        url = f"{Constants.IFLOW_BASE_URL}/chat/completions"
        headers = {
            'Authorization': f'Bearer {credentials.apiKey}', # <--- ИСПОЛЬЗУЕМ API KEY
            'Content-Type': 'application/json'
        }
        response = await http_client.post(url, headers=headers, json=request_body, timeout=Constants.REQUEST_TIMEOUT)
        logger.debug(f"[{request_id}] API response: {response.text}")
        return response

    async def _retry_with_refresh(self, http_client: httpx.AsyncClient, credentials: OAuthCredentials, creds_path: str, request_body: Dict[str, Any], request_id: str) -> httpx.Response:
        logger.warning(f"[{request_id}] Got 401 or missing apiKey, attempting token refresh")
        credentials = await self.cred_manager._refresh_with_deduplication(credentials, creds_path)
        response = await self._execute_request(http_client, credentials, request_body, request_id)
        logger.info(f"[{request_id}] Retry after refresh status: {response.status_code}")
        return response

    @staticmethod
    async def _read_error_response(response: httpx.Response) -> str:
        try:
            return (await response.aread()).decode()
        except Exception:
            return "[Could not read error response]"


# ===========================
# FastAPI Application
# ===========================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.account_manager = AccountManager(Settings.IFLOW_ACCOUNTS, Settings.ACCOUNT_STATE_FILE, Settings.IFLOW_INITIAL_BACKOFF)
    app.state.credential_manager = CredentialManager()
    app.state.api_client = ApiClient(app.state.account_manager, app.state.credential_manager)
    app.state.http_client = httpx.AsyncClient()
    logger.info(f"Server started with {len(Settings.IFLOW_ACCOUNTS)} iFlow account(s)")
    if Settings.IFLOW_ACCOUNTS:
        await verify_accounts(app.state.credential_manager)
    logger.info("Proxy is ready!")
    yield
    await app.state.credential_manager.close()
    await app.state.http_client.aclose()


async def verify_api_key(authorization: Optional[str] = Header(None)) -> None:
    if Settings.API_KEY and (not authorization or not authorization.startswith("Bearer ") or authorization[7:] != Settings.API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")


app = FastAPI(title="iFlow Multi-Account Proxy", description="OpenAI-compatible proxy for iFlow API", lifespan=lifespan)


async def verify_accounts(cred_manager: CredentialManager):
    logger.info("--- Verifying Account Credentials ---")
    for path in Settings.IFLOW_ACCOUNTS:
        try:
            creds = await cred_manager.get_credentials(path)
            expiry_dt = datetime.fromtimestamp(creds.expiry_date / 1000)
            logger.info(f"  -> {path} | SUCCESS | Token expires: {expiry_dt}")
        except Exception as e:
            logger.error(f"  -> {path} | FAILED | Error: {e}")
    logger.info("--- Verification Complete ---")


# ===========================
# API Endpoints
# ===========================

@app.get("/iflow/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    return {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "iflow"} for model in Settings.IFLOW_AVAILABLE_MODELS]}


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def create_chat_completion(request: Request):
    request_id = f"iflow-chatcmpl-{uuid.uuid4()}"
    try:
        openai_request_body = await request.json()
        logger.info(f"[{request_id}] Received request for model {openai_request_body.get('model')}.")
        client: ApiClient = request.app.state.api_client
        response_data = await client.chat_completion(request, openai_request_body, request_id)
        return JSONResponse(content=response_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Error in endpoint")
        raise HTTPException(status_code=500, detail=str(e))


# ===========================
# Health Check
# ===========================

@app.get("/health")
async def health_check():
    return {"status": "healthy", "iflow_accounts": len(Settings.IFLOW_ACCOUNTS), "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)