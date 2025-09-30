import asyncio
import itertools
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, AsyncGenerator, Union

import aiofiles
import httpx
import uvicorn
import yaml
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse, JSONResponse
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from loguru import logger


# ===========================
# Constants
# ===========================

class Constants:
    """Centralized constants for configuration values."""

    # Token validity buffers
    GEMINI_TOKEN_BUFFER_MINUTES = 5

    # HTTP retry configuration
    MAX_RETRIES = 3
    BASE_RETRY_DELAY = 1.0

    # HTTP timeouts
    REQUEST_TIMEOUT = 300.0
    DEFAULT_TIMEOUT = 30.0

    # Connection limits
    MAX_KEEPALIVE_CONNECTIONS = 10
    MAX_CONNECTIONS = 50


    # Gemini-specific
    GEMINI_TEMPERATURE = 0.05
    GEMINI_MAX_OUTPUT_TOKENS = 64000
    GEMINI_TOP_P = 0.95
    GEMINI_THINKING_BUDGET_PRO = 32768
    GEMINI_THINKING_BUDGET_DEFAULT = 24576

    # Error response truncation
    ERROR_RESPONSE_MAX_LENGTH = 500
    ERROR_TEXT_MAX_LENGTH = 1000

    # Rate limit status codes
    RATE_LIMIT_STATUS_CODES = {429, 500}


# ===========================
# Configuration Management
# ===========================

class Settings:
    """Application settings loaded from config.yaml."""

    @staticmethod
    def load_config() -> Dict[str, Any]:
        """Load configuration from config.yaml file."""
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

    # Gemini Settings
    GEMINI_OAUTH_CLIENT_ID: str = config.get("gemini", {}).get("oauth", {}).get("client_id", "")
    GEMINI_OAUTH_CLIENT_SECRET: str = config.get("gemini", {}).get("oauth", {}).get("client_secret", "")
    GEMINI_ACCOUNTS: Dict[str, str] = config.get("gemini", {}).get("accounts", {})
    GEMINI_DEFAULT_MODEL: str = config.get("gemini", {}).get("default_model", "gemini-2.5-pro")
    GEMINI_AVAILABLE_MODELS: List[str] = config.get("gemini", {}).get("available_models", ["gemini-2.5-pro"])
    GEMINI_INITIAL_BACKOFF: int = config.get("gemini", {}).get("initial_backoff_seconds", 30)

    # Shared Settings
    CODE_ASSIST_ENDPOINT: str = "https://cloudcode-pa.googleapis.com"
    CODE_ASSIST_API_VERSION: str = "v1internal"
    TOKEN_URI: str = "https://oauth2.googleapis.com/token"
    USERINFO_ENDPOINT: str = "https://www.googleapis.com/oauth2/v3/userinfo"

    GEMINI_ACCOUNT_STATE_FILE: str = "gemini_account_states.json"


logger.remove()
logger.add(sys.stderr, level="DEBUG")


# ===========================
# Utility Functions
# ===========================

async def read_json_file(path: str) -> Dict[str, Any]:
    """Read and parse a JSON file asynchronously."""
    async with aiofiles.open(path, 'r') as f:
        content = await f.read()
        return json.loads(content)


async def write_json_file(path: str, data: Dict[str, Any]) -> None:
    """Write data to a JSON file asynchronously."""
    async with aiofiles.open(path, 'w') as f:
        await f.write(json.dumps(data, indent=2))


def calculate_exponential_backoff(failure_count: int, initial_backoff: int) -> float:
    """Calculate exponential backoff duration based on failure count."""
    return initial_backoff * (2 ** (failure_count - 1))


def is_rate_limit_error(status_code: int) -> bool:
    """Check if HTTP status code indicates rate limiting."""
    return status_code in Constants.RATE_LIMIT_STATUS_CODES


# ===========================
# Gemini Credential Management
# ===========================

class GeminiTokenValidator:
    """Validates Gemini token expiry."""

    @staticmethod
    def is_token_expired(credentials: Credentials) -> bool:
        """Check if Gemini token is expired with buffer time."""
        return (
                not credentials.expiry
                or credentials.expiry.replace(tzinfo=None) <
                datetime.utcnow() + timedelta(minutes=Constants.GEMINI_TOKEN_BUFFER_MINUTES)
        )


class GeminiCredentialFactory:
    """Creates Credentials objects from file data."""

    @staticmethod
    def create_from_file_info(file_info: Dict[str, Any]) -> Credentials:
        """Create Credentials object from file data."""
        expiry_utc_string = GeminiCredentialFactory._format_expiry(file_info)
        scopes = GeminiCredentialFactory._parse_scopes(file_info)

        auth_info = {
            "token": file_info.get('access_token'),
            "refresh_token": file_info.get('refresh_token'),
            "id_token": file_info.get('id_token'),
            "token_uri": Settings.TOKEN_URI,
            "client_id": Settings.GEMINI_OAUTH_CLIENT_ID,
            "client_secret": Settings.GEMINI_OAUTH_CLIENT_SECRET,
            "scopes": scopes,
            "expiry": expiry_utc_string
        }
        return Credentials.from_authorized_user_info(auth_info)

    @staticmethod
    def _format_expiry(file_info: Dict[str, Any]) -> Optional[str]:
        """Format expiry date from file info."""
        if 'expiry_date' in file_info and file_info['expiry_date']:
            expiry_dt = datetime.fromtimestamp(file_info['expiry_date'] / 1000, tz=timezone.utc)
            return expiry_dt.isoformat().replace('+00:00', 'Z')
        return None

    @staticmethod
    def _parse_scopes(file_info: Dict[str, Any]) -> Optional[List[str]]:
        """Parse scopes from file info."""
        scope = file_info.get('scope')
        return scope.split() if isinstance(scope, str) else scope


class GeminiCredentialManager:
    """Manages Gemini OAuth credentials with caching and refresh."""

    def __init__(self) -> None:
        self._credentials_cache: Dict[str, Credentials] = {}
        self._refresh_locks: Dict[str, asyncio.Lock] = {}
        self.validator = GeminiTokenValidator()
        self.factory = GeminiCredentialFactory()

    async def get_credentials(self, creds_path: str) -> Credentials:
        """Get valid credentials, refreshing if necessary."""
        expanded_path = self._expand_path(creds_path)

        if expanded_path not in self._refresh_locks:
            self._refresh_locks[expanded_path] = asyncio.Lock()

        async with self._refresh_locks[expanded_path]:
            if self._has_valid_cached_credentials(expanded_path):
                return self._credentials_cache[expanded_path]

            file_info = await self._load_credential_file(expanded_path)
            credentials = self.factory.create_from_file_info(file_info)

            if self.validator.is_token_expired(credentials) and credentials.refresh_token:
                await self._refresh_and_save(credentials, file_info, expanded_path)

            self._credentials_cache[expanded_path] = credentials

            if not credentials.token:
                raise HTTPException(
                    status_code=500,
                    detail=f"No valid access token for {expanded_path}"
                )

            return credentials

    @staticmethod
    def _expand_path(creds_path: str) -> str:
        """Expand path, handling both absolute and relative paths."""
        if not creds_path.startswith('/') and not creds_path.startswith('~'):
            return creds_path
        return os.path.expanduser(creds_path)

    def _has_valid_cached_credentials(self, path: str) -> bool:
        """Check if valid credentials exist in cache."""
        return (path in self._credentials_cache and
                not self.validator.is_token_expired(self._credentials_cache[path]))

    @staticmethod
    async def _load_credential_file(path: str) -> Dict[str, Any]:
        """Load credential file with error handling."""
        try:
            return await read_json_file(path)
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail=f"Credentials file not found: {path}")
        except (json.JSONDecodeError, KeyError) as e:
            raise HTTPException(status_code=500, detail=f"Credentials file malformed: {e}")

    async def _refresh_and_save(
            self,
            credentials: Credentials,
            file_info: Dict[str, Any],
            path: str
    ) -> None:
        """Refresh credentials and save to file."""
        logger.info(f"Token for {path} requires refresh.")
        try:
            await asyncio.to_thread(credentials.refresh, GoogleAuthRequest())

            file_info['access_token'] = credentials.token
            file_info['refresh_token'] = credentials.refresh_token
            if credentials.expiry:
                file_info['expiry_date'] = int(credentials.expiry.timestamp() * 1000)

            await write_json_file(path, file_info)
            logger.info(f"Token for {path} refreshed and saved.")
        except Exception as e:
            self._credentials_cache.pop(path, None)
            raise HTTPException(status_code=500, detail=f"Failed to refresh token for {path}: {e}")


# ===========================
# Account Management
# ===========================

class AccountState(Enum):
    """States an account can be in."""
    AVAILABLE = "available"
    BLOCKED = "blocked"


class Account:
    """Represents an API account with rate limit state."""

    def __init__(self, creds_path: str, project_name: Optional[str] = None):
        self.creds_path = creds_path
        self.project_name = project_name
        self.status: AccountState = AccountState.AVAILABLE
        self.failure_count: int = 0
        self.available_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize account to dictionary."""
        data = {
            "creds_path": self.creds_path,
            "status": self.status.value,
            "failure_count": self.failure_count,
            "available_at": self.available_at,
        }
        if self.project_name:
            data["project_name"] = self.project_name
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Account":
        """Deserialize account from dictionary."""
        account = cls(data["creds_path"], data.get("project_name"))
        account.status = AccountState(data.get("status", "available"))
        account.failure_count = data.get("failure_count", 0)
        account.available_at = data.get("available_at", 0.0)
        return account


class AccountManager:
    """Manages multiple accounts with rotation and rate limiting."""

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
        """Initialize accounts from dictionary configuration."""
        for path, proj in accounts_config.items():
            self.accounts[path] = Account(path, proj)

    def _init_from_list(self, accounts_config: List[str]):
        """Initialize accounts from list configuration."""
        for path in accounts_config:
            self.accounts[path] = Account(path)

    def _load_states(self):
        """Load account states from disk."""
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
        """Save account states to disk."""
        try:
            states = {path: acc.to_dict() for path, acc in self.accounts.items()}
            await write_json_file(self.state_file, states)
        except IOError as e:
            logger.error(f"Failed to save account states: {e}")

    async def get_available_account(self) -> Optional[Account]:
        """Get next available account, checking cooldown status."""
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
        """Get next account path from cycler."""
        next_item = next(self.account_cycler)
        if isinstance(next_item, tuple):
            self.account_cycler = itertools.cycle(self.accounts.keys())
            return next(self.account_cycler)
        return next_item

    @staticmethod
    def _is_account_in_cooldown(account: Account) -> bool:
        """Check if account is in cooldown period."""
        return account.status == AccountState.BLOCKED and time.time() < account.available_at

    @staticmethod
    def _mark_account_available(account: Account) -> None:
        """Mark a blocked account as available after cooldown."""
        logger.info(f"Account {account.creds_path} is now available after cooldown.")
        account.status = AccountState.AVAILABLE
        account.failure_count = 0

    async def report_failure(self, creds_path: str):
        """Report account failure and apply exponential backoff."""
        async with self._lock:
            if creds_path in self.accounts:
                account = self.accounts[creds_path]
                account.status = AccountState.BLOCKED
                account.failure_count += 1
                backoff_duration = calculate_exponential_backoff(
                    account.failure_count, 
                    self.initial_backoff
                )
                account.available_at = time.time() + backoff_duration
                logger.warning(
                    f"Account {creds_path} hit rate limit. "
                    f"Failure count: {account.failure_count}. "
                    f"Blocking for {backoff_duration} seconds."
                )
                await self._save_states()


# ===========================
# Gemini API Client
# ===========================

class GeminiMessageTransformer:
    """Transforms OpenAI format messages to Gemini format."""
    
    @staticmethod
    def transform_messages(openai_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI messages to Gemini format."""
        messages = openai_data.get("messages", [])
        system_prompt = GeminiMessageTransformer._extract_system_prompt(messages)
        gemini_contents = GeminiMessageTransformer._convert_messages(messages)
        
        if system_prompt:
            GeminiMessageTransformer._inject_system_prompt(gemini_contents, system_prompt)
        
        if gemini_contents and gemini_contents[0]["role"] == "model":
            gemini_contents.insert(0, {"role": "user", "parts": [{"text": ""}]})
        
        return gemini_contents

    @staticmethod
    def _extract_system_prompt(messages: List[Dict[str, Any]]) -> str:
        """Extract system prompt from messages if present."""
        if messages and messages[0].get('role') == 'system':
            return messages.pop(0).get('content', '')
        return ""

    @staticmethod
    def _convert_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert message list to Gemini format."""
        gemini_contents = []
        for message in messages:
            role = "model" if message.get("role") == "assistant" else "user"
            content = GeminiMessageTransformer._extract_content(message)
            gemini_contents.append({"role": role, "parts": [{"text": content}]})
        return gemini_contents

    @staticmethod
    def _extract_content(message: Dict[str, Any]) -> str:
        """Extract text content from message."""
        raw_content = message.get("content", "")
        if isinstance(raw_content, list):
            return "\n".join(
                part.get("text", "") 
                for part in raw_content 
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return raw_content

    @staticmethod
    def _inject_system_prompt(gemini_contents: List[Dict[str, Any]], system_prompt: str) -> None:
        """Inject system prompt into first user message."""
        try:
            first_user_idx = next(
                i for i, msg in enumerate(gemini_contents) 
                if msg['role'] == 'user'
            )
            original_text = gemini_contents[first_user_idx]['parts'][0]['text']
            gemini_contents[first_user_idx]['parts'][0]['text'] = f"{system_prompt}\n\n{original_text}"
        except StopIteration:
            gemini_contents.insert(0, {"role": "user", "parts": [{"text": system_prompt}]})


class GeminiRequestBuilder:
    """Builds Gemini API requests."""
    
    @staticmethod
    def build_request(openai_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI request to Gemini format."""
        model = openai_data.get("model", Settings.GEMINI_DEFAULT_MODEL)
        gemini_contents = GeminiMessageTransformer.transform_messages(openai_data)
        
        return {
            "model": model,
            "request": {
                "contents": gemini_contents,
                "generationConfig": GeminiRequestBuilder._build_generation_config(model),
                "tools": [{"googleSearch": {}}],
            },
        }

    @staticmethod
    def _build_generation_config(model: str) -> Dict[str, Any]:
        """Build generation configuration for model."""
        thinking_budget = (
            Constants.GEMINI_THINKING_BUDGET_PRO 
            if model == "gemini-2.5-pro" 
            else Constants.GEMINI_THINKING_BUDGET_DEFAULT
        )
        
        return {
            "thinkingConfig": {"thinkingBudget": thinking_budget},
            "temperature": Constants.GEMINI_TEMPERATURE,
            "maxOutputTokens": Constants.GEMINI_MAX_OUTPUT_TOKENS,
            "topP": Constants.GEMINI_TOP_P,
        }


class GeminiResponseProcessor:
    """Processes Gemini API responses."""
    
    @staticmethod
    def extract_text_from_chunk(chunk: Dict[str, Any]) -> str:
        """Extract text content from response chunk."""
        try:
            data = chunk.get('response', chunk)
            return data['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError, TypeError):
            return ""

    @staticmethod
    def extract_finish_reason(chunk: Dict[str, Any]) -> Optional[str]:
        """Extract and map finish reason from chunk."""
        data = chunk.get('response', chunk)
        candidates = data.get('candidates', [{}])
        finish_reason_raw = candidates[0].get('finishReason') if candidates else None
        
        if not finish_reason_raw:
            return None
        
        finish_reason_mapping = {
            'STOP': 'stop',
            'MAX_TOKENS': 'length',
            'SAFETY': 'content_filter',
            'RECITATION': 'content_filter',
            'OTHER': 'stop'
        }
        return finish_reason_mapping.get(finish_reason_raw, 'stop')

    @staticmethod
    def extract_usage_metadata(chunk: Dict[str, Any]) -> Dict[str, int]:
        """Extract usage metadata from chunk."""
        data = chunk.get('response', chunk)
        usage_metadata = data.get("usageMetadata", {})
        
        return {
            "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
            "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
            "total_tokens": usage_metadata.get("totalTokenCount", 0)
        }


class GeminiApiClient:
    """Client for Gemini API operations."""
    
    def __init__(self, account_manager: AccountManager, cred_manager: GeminiCredentialManager) -> None:
        self.account_manager = account_manager
        self.cred_manager = cred_manager
        self.request_builder = GeminiRequestBuilder()
        self.response_processor = GeminiResponseProcessor()

    async def stream_chat(
        self, 
        request: Request, 
        request_body: Dict[str, Any], 
        request_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat completion with automatic account rotation."""
        num_accounts = len(self.account_manager.accounts)

        for attempt in range(num_accounts):
            account = await self.account_manager.get_available_account()
            if not account:
                logger.error(f"[{request_id}] No available Gemini accounts.")
                break

            logger.info(f"[{request_id}] Gemini attempt {attempt + 1}/{num_accounts} with: {account.creds_path}")

            try:
                async for chunk in self._stream_from_account(request, request_body, account, request_id):
                    yield chunk
                return

            except httpx.HTTPStatusError as e:
                if is_rate_limit_error(e.response.status_code):
                    await self.account_manager.report_failure(account.creds_path)
                    logger.warning(f"[{request_id}] Rate limit hit. Trying next Gemini account.")
                    continue
                else:
                    logger.exception(f"[{request_id}] Unrecoverable HTTP error for Gemini.")
                    yield {"error": {"message": f"HTTP error: {str(e)}"}}
                    return
            except Exception as e:
                logger.exception(f"[{request_id}] Unexpected error for Gemini account.")
                yield {"error": {"message": f"Unexpected error: {str(e)}"}}
                return

        logger.error(f"[{request_id}] All {num_accounts} Gemini accounts failed.")
        yield {"error": {"message": "All Gemini accounts are rate-limited."}}

    async def _stream_from_account(
        self,
        request: Request,
        request_body: Dict[str, Any],
        account: Account,
        request_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream responses from a specific account."""
        http_client: httpx.AsyncClient = request.app.state.http_client
        credentials = await self.cred_manager.get_credentials(account.creds_path)

        headers = {
            'Authorization': f'Bearer {credentials.token}',
            'Content-Type': 'application/json'
        }

        url = f"{Settings.CODE_ASSIST_ENDPOINT}/{Settings.CODE_ASSIST_API_VERSION}:streamGenerateContent"
        request_body['project'] = account.project_name

        async with http_client.stream(
            "POST", 
            url, 
            headers=headers, 
            json=request_body, 
            timeout=Constants.REQUEST_TIMEOUT,
            params={"alt": "sse"}
        ) as response:
            logger.info(f"[{request_id}] Gemini API status: {response.status_code}")

            if response.status_code == 400:
                logger.error(f"[{request_id}] Bad request. Response: {await response.aread()}")

            if response.status_code == 500:
                logger.error(f"[{request_id}] Server error. Response: {await response.aread()}")

            response.raise_for_status()
            logger.info(f"[{request_id}] Successfully connected with Gemini account.")

            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning(f"[{request_id}] Could not decode JSON chunk: {line}")


# ===========================
# FastAPI Application
# ===========================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Initialize Gemini components
    app.state.gemini_account_manager = AccountManager(
        Settings.GEMINI_ACCOUNTS,
        Settings.GEMINI_ACCOUNT_STATE_FILE,
        Settings.GEMINI_INITIAL_BACKOFF
    )
    app.state.gemini_credential_manager = GeminiCredentialManager()
    app.state.gemini_client = GeminiApiClient(
        app.state.gemini_account_manager,
        app.state.gemini_credential_manager
    )

    # Shared HTTP client
    app.state.http_client = httpx.AsyncClient()

    logger.info(
        f"Server started with {len(Settings.GEMINI_ACCOUNTS)} Gemini account(s) "
    )

    # Verify accounts
    if Settings.GEMINI_ACCOUNTS:
        await verify_gemini_accounts(app.state.gemini_credential_manager)

    logger.info("Proxy is ready!")

    yield

    await app.state.http_client.aclose()


async def verify_api_key(authorization: Optional[str] = Header(None)) -> None:
    """Verify API key from authorization header."""
    if Settings.API_KEY and (
            not authorization or 
            not authorization.startswith("Bearer ") or 
            authorization[7:] != Settings.API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")


app = FastAPI(
    title="Gemini Multi-Account Proxy",
    description="OpenAI-compatible proxy for both Gemini API",
    lifespan=lifespan
)


async def verify_gemini_accounts(cred_manager: GeminiCredentialManager):
    """Verify Gemini account credentials on startup."""
    logger.info("--- Verifying Gemini Account Credentials ---")
    for creds_path, project_name in Settings.GEMINI_ACCOUNTS.items():
        try:
            credentials = await cred_manager.get_credentials(creds_path)
            headers = {'Authorization': f'Bearer {credentials.token}'}
            async with httpx.AsyncClient() as client:
                response = await client.get(Settings.USERINFO_ENDPOINT, headers=headers)
            if response.status_code == 200:
                email = response.json().get('email', 'N/A')
                logger.info(f"  -> {creds_path} | {project_name} | {email} | Valid")
            else:
                logger.warning(f"  -> {creds_path} | {project_name} | Failed | Status: {response.status_code}")
        except Exception as e:
            logger.error(f"  -> {creds_path} | Error: {e}")
    logger.info("--- Gemini Verification Complete ---")



# ===========================
# Gemini Endpoints
# ===========================

@app.get("/gemini/v1/models", dependencies=[Depends(verify_api_key)])
async def list_gemini_models() -> Dict[str, Any]:
    """List available Gemini models."""
    return {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "owned_by": "google"} 
            for model in Settings.GEMINI_AVAILABLE_MODELS
        ]
    }


class GeminiStreamResponseGenerator:
    """Generates OpenAI-compatible streaming responses from Gemini."""
    
    def __init__(self, client: GeminiApiClient, processor: GeminiResponseProcessor):
        self.client = client
        self.processor = processor

    async def generate(
        self,
        request: Request,
        gemini_request: Dict[str, Any],
        request_id: str,
        model: str
    ) -> AsyncGenerator[str, None]:
        """Generate SSE-formatted streaming response."""
        created_time = int(time.time())
        has_sent_initial_chunk = False

        try:
            async for chunk in self.client.stream_chat(request, gemini_request, request_id):
                if "error" in chunk:
                    yield self._format_error_chunk(chunk, request_id)
                    return

                if not has_sent_initial_chunk:
                    yield self._format_initial_chunk(request_id, created_time, model)
                    has_sent_initial_chunk = True

                openai_chunk = self._convert_chunk_to_openai(
                    chunk, 
                    request_id, 
                    created_time, 
                    model
                )
                yield f"data: {json.dumps(openai_chunk)}\n\n"

                if openai_chunk["choices"][0]["finish_reason"]:
                    break

        except Exception as e:
            yield self._format_error_chunk(
                {"error": {"message": str(e)}}, 
                request_id
            )
            logger.exception(f"[{request_id}] Unexpected error in Gemini stream: {str(e)}")

        yield "data: [DONE]\n\n"
        logger.info(f"[{request_id}] Gemini stream finished.")

    @staticmethod
    def _format_error_chunk(chunk: Dict[str, Any], request_id: str) -> str:
        """Format error chunk as SSE."""
        error_detail = chunk["error"].get("message", "Unknown error")
        error_response = {
            "error": {
                "message": error_detail,
                "type": "server_error",
                "code": "internal_error"
            }
        }
        logger.error(f"[{request_id}] Error during Gemini stream: {error_detail}")
        return f"data: {json.dumps(error_response)}\n\n"

    @staticmethod
    def _format_initial_chunk(request_id: str, created_time: int, model: str) -> str:
        """Format initial chunk with role assignment."""
        initial_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
                "logprobs": None
            }],
            "system_fingerprint": None
        }
        return f"data: {json.dumps(initial_chunk)}\n\n"

    def _convert_chunk_to_openai(
        self,
        chunk: Dict[str, Any],
        request_id: str,
        created_time: int,
        model: str
    ) -> Dict[str, Any]:
        """Convert Gemini chunk to OpenAI format."""
        text_chunk = self.processor.extract_text_from_chunk(chunk)
        finish_reason = self.processor.extract_finish_reason(chunk)

        delta = {}
        if text_chunk:
            delta["content"] = text_chunk

        openai_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
                "logprobs": None
            }],
            "system_fingerprint": None
        }

        if finish_reason:
            usage = self.processor.extract_usage_metadata(chunk)
            if usage:
                openai_chunk["usage"] = usage

        return openai_chunk


async def _handle_gemini_stream_request(
    request: Request, 
    gemini_request: Dict[str, Any],
    request_id: str
) -> StreamingResponse:
    """Handle streaming Gemini request."""
    client: GeminiApiClient = request.app.state.gemini_client
    model = gemini_request.get("model", Settings.GEMINI_DEFAULT_MODEL)
    
    processor = GeminiResponseProcessor()
    generator = GeminiStreamResponseGenerator(client, processor)

    return StreamingResponse(
        generator.generate(request, gemini_request, request_id, model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


async def _handle_gemini_non_stream_request(
    request: Request, 
    gemini_request: Dict[str, Any],
    request_id: str
) -> JSONResponse:
    """Handle non-streaming Gemini request."""
    client: GeminiApiClient = request.app.state.gemini_client
    processor = GeminiResponseProcessor()
    model = gemini_request.get("model", Settings.GEMINI_DEFAULT_MODEL)

    full_response_text: List[str] = []
    last_chunk: Dict[str, Any] = {}

    async for chunk in client.stream_chat(request, gemini_request, request_id):
        if error := chunk.get("error"):
            raise HTTPException(
                status_code=502, 
                detail=error.get("message", "Upstream service error")
            )
        full_response_text.append(processor.extract_text_from_chunk(chunk))
        last_chunk = chunk

    usage = processor.extract_usage_metadata(last_chunk)

    final_response = {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "".join(full_response_text).strip()
            },
            "finish_reason": "stop"
        }],
        "usage": usage
    }

    logger.info(f"[{request_id}] Gemini non-stream request finished.")
    return JSONResponse(content=final_response)


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def create_gemini_chat_completion(request: Request):
    """Create Gemini chat completion (streaming or non-streaming)."""
    request_id = f"gemini-chatcmpl-{uuid.uuid4()}"

    try:
        incoming_data = await request.json()
        logger.info(f"[{request_id}] Received Gemini request. Stream: {incoming_data.get('stream', False)}")

        request_builder = GeminiRequestBuilder()
        gemini_request = request_builder.build_request(incoming_data)

        if incoming_data.get("stream", False):
            return await _handle_gemini_stream_request(request, gemini_request, request_id)
        else:
            return await _handle_gemini_non_stream_request(request, gemini_request, request_id)

    except Exception as e:
        logger.exception(f"[{request_id}] Error in Gemini endpoint")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))


# ===========================
# Health Check
# ===========================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "gemini_accounts": len(Settings.GEMINI_ACCOUNTS),
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)