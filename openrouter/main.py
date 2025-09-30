#!/usr/bin/env python3
"""
OpenRouter API Proxy
Proxies requests to OpenRouter API and rotates API keys to bypass rate limits.
"""

import asyncio
import json
import logging
import random
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional, Union

import httpx
import uvicorn
import yaml
from fastapi import APIRouter, Request, Header, FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response, JSONResponse

# Constants
CONFIG_FILE = "config.yaml"
RATE_LIMIT_ERROR_CODE = 429
SERVER_ERROR_CODE = 500
MAX_RETRY_ATTEMPTS = 6
MODELS_ENDPOINTS = ["/api/v1/models"]
GLOBAL_LIMIT_PATTERN = "is temporarily rate-limited upstream"
GOOGLE_LIMIT_ERROR = "Google returned RESOURCE_EXHAUSTED code"
GLOBAL_LIMIT_ERROR = "Model is temporarily rate-limited upstream"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_PUBLIC_ENDPOINTS = ["/api/v1/models"]
DEFAULT_KEY_SELECTION_STRATEGY = "round-robin"
DEFAULT_FREE_ONLY = False
DEFAULT_GLOBAL_RATE_DELAY = 0
DEFAULT_PROXY_ENABLED = False
DEFAULT_PROXY_URL = ""
DEFAULT_HTTP_TIMEOUT = 600.0
MIN_BODY_SIZE_FOR_CHECK = 10
MAX_BODY_SIZE_FOR_CHECK = 4000
KEY_MASK_LENGTH = 4


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(CONFIG_FILE, encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Configuration file {CONFIG_FILE} not found. "
              "Please create it based on config.yaml.example.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)


def setup_logging(config_: Dict[str, Any]) -> logging.Logger:
    """Configure logging based on configuration."""
    log_level_str = config_.get("server", {}).get("log_level", "INFO")
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger_ = logging.getLogger("openrouter-proxy")
    logger_.info("Logging level set to %s", log_level_str)

    return logger_


def normalize_and_validate_config(config_data: Dict[str, Any]) -> None:
    """
    Normalizes the configuration by adding defaults for missing keys
    and validates the structure and types, logging warnings/errors.
    Modifies the config_data dictionary in place.
    """
    _normalize_openrouter_config(config_data)


def _normalize_openrouter_config(config_data: Dict[str, Any]) -> None:
    """Normalize and validate OpenRouter configuration section."""
    if not isinstance(config_data.get("openrouter"), dict):
        logger.warning("'openrouter' section missing or invalid in config.yaml. Using defaults.")
        config_data["openrouter"] = {}

    openrouter_config = config_data["openrouter"]

    # Base URL
    if not isinstance(openrouter_config.get("base_url"), str):
        logger.warning(
            "'openrouter.base_url' missing or invalid in config.yaml. Using default: %s",
            DEFAULT_BASE_URL
        )
        openrouter_config["base_url"] = DEFAULT_BASE_URL
    openrouter_config["base_url"] = openrouter_config["base_url"].rstrip("/")

    # Public endpoints
    if "public_endpoints" in openrouter_config and openrouter_config["public_endpoints"] is None:
        openrouter_config["public_endpoints"] = []

    if not isinstance(openrouter_config.get("public_endpoints"), list):
        logger.warning(
            "'openrouter.public_endpoints' missing or invalid in config.yaml. Using default: %s",
            DEFAULT_PUBLIC_ENDPOINTS
        )
        openrouter_config["public_endpoints"] = DEFAULT_PUBLIC_ENDPOINTS
    else:
        openrouter_config["public_endpoints"] = _validate_endpoints(
            openrouter_config["public_endpoints"]
        )

    # API keys
    if not isinstance(openrouter_config.get("keys"), list):
        logger.warning("'openrouter.keys' missing or invalid in config.yaml. Using empty list.")
        openrouter_config["keys"] = []

    if not openrouter_config["keys"]:
        logger.warning(
            "'openrouter.keys' list is empty in config.yaml. "
            "Proxy will not work for authenticated endpoints."
        )

    # Key selection strategy
    key_selection_strategy = openrouter_config.get("key_selection_strategy")
    valid_strategies = ["round-robin", "first", "random"]

    if (not isinstance(key_selection_strategy, str) or
            key_selection_strategy not in valid_strategies):
        logger.warning(
            "'openrouter.key_selection_strategy' is unknown: '%s', set '%s'",
            str(key_selection_strategy), DEFAULT_KEY_SELECTION_STRATEGY
        )
        openrouter_config["key_selection_strategy"] = DEFAULT_KEY_SELECTION_STRATEGY

    # Key selection options
    if not isinstance(openrouter_config.get("key_selection_opts"), list):
        logger.warning(
            "'openrouter.key_selection_opts' missing or invalid in config.yaml. Using empty list."
        )
        openrouter_config["key_selection_opts"] = []

    # Free only flag
    if not isinstance(openrouter_config.get("free_only"), bool):
        logger.warning(
            "'openrouter.free_only' missing or invalid in config.yaml. Using default: %s",
            DEFAULT_FREE_ONLY
        )
        openrouter_config["free_only"] = DEFAULT_FREE_ONLY

    # Global rate delay
    if not isinstance(openrouter_config.get("global_rate_delay"), (int, float)):
        logger.warning(
            "'openrouter.global_rate_delay' missing or invalid in config.yaml. Using default: %s",
            DEFAULT_GLOBAL_RATE_DELAY
        )
        openrouter_config["global_rate_delay"] = DEFAULT_GLOBAL_RATE_DELAY


def _validate_endpoints(endpoints: List[Any]) -> List[str]:
    """Validate and normalize endpoint list."""
    validated_endpoints = []

    for i, endpoint in enumerate(endpoints):
        if not isinstance(endpoint, str):
            logger.warning(
                "Item %d in 'openrouter.public_endpoints' is not a string. Skipping.", i
            )
            continue

        if not endpoint:
            logger.warning(
                "Item %d in 'openrouter.public_endpoints' is empty. Skipping.", i
            )
            continue

        # Ensure leading slash
        normalized = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        validated_endpoints.append(normalized)

    return validated_endpoints



# Load configuration
config = load_config()

# Initialize logging
logger = setup_logging(config)

# Normalize and validate configuration (modifies config in place)
normalize_and_validate_config(config)


def mask_key(key: str) -> str:
    """Mask an API key for logging purposes."""
    if not key:
        return key
    if len(key) <= 8:
        return "****"
    return f"{key[:KEY_MASK_LENGTH]}****{key[-KEY_MASK_LENGTH:]}"


class KeyManager:
    """Manages OpenRouter API keys, including rotation and rate limit handling."""

    def __init__(
            self,
            keys: List[str],
            cooldown_seconds: int,
            strategy: str,
            opts: List[str]
    ):
        """
        Initialize the KeyManager.

        Args:
            keys: List of API keys to manage
            cooldown_seconds: Time in seconds to disable a key after rate limit
            strategy: Key selection strategy ('round-robin', 'first', or 'random')
            opts: Additional options for key selection
        """
        self.keys = keys
        self.cooldown_seconds = cooldown_seconds
        self.current_index = 0
        self.disabled_until: Dict[str, datetime] = {}
        self.strategy = strategy
        self.use_last_key = "same" in opts
        self.last_key: Optional[str] = None
        self.lock = asyncio.Lock()

        if not keys:
            logger.error("No API keys provided in configuration.")
            sys.exit(1)

    async def get_next_key(self) -> str:
        """
        Get the next available API key using the configured selection strategy.

        Returns:
            The selected API key

        Raises:
            HTTPException: If all keys are currently disabled
        """
        async with self.lock:
            available_keys = self._get_available_keys()

            if not available_keys:
                return await self._handle_no_available_keys()

            selected_key = self._select_key(available_keys)
            self.last_key = selected_key
            return selected_key

    def _get_available_keys(self) -> List[str]:
        """Get list of currently available (not disabled) keys."""
        available_keys = []
        now = datetime.now()

        for key in self.keys:
            if key in self.disabled_until:
                if now >= self.disabled_until[key]:
                    # Key cooldown period has expired
                    del self.disabled_until[key]
                    logger.info("API key %s is now enabled again.", mask_key(key))
                    available_keys.append(key)
            else:
                # Key is not disabled
                available_keys.append(key)

        return available_keys

    async def _handle_no_available_keys(self) -> str:
        """Handle the case when all keys are disabled."""
        now = datetime.now()
        soonest_available = min(self.disabled_until.values())
        wait_seconds = (soonest_available - now).total_seconds()

        logger.error(
            "All API keys are currently disabled. The next key will be available in %.2f seconds.",
            wait_seconds
        )

        raise HTTPException(
            status_code=503,
            detail="All API keys are currently disabled due to rate limits. Please try again later."
        )

    def _select_key(self, available_keys: List[str]) -> str:
        """
        Select a key from available keys based on the configured strategy.

        Args:
            available_keys: List of available keys

        Returns:
            The selected key
        """
        available_keys_set = set(available_keys)

        # Check if we should use the last key
        if self.use_last_key and self.last_key in available_keys_set:
            return self.last_key

        if self.strategy == "round-robin":
            return self._select_round_robin(available_keys_set)
        elif self.strategy == "first":
            return available_keys[0]
        elif self.strategy == "random":
            return random.choice(available_keys)
        else:
            raise RuntimeError(f"Unknown key selection strategy: {self.strategy}")

    def _select_round_robin(self, available_keys_set: set) -> str:
        """Select key using round-robin strategy."""
        for _ in self.keys:
            key = self.keys[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.keys)
            if key in available_keys_set:
                return key
        # This should never happen if available_keys_set is not empty
        raise RuntimeError("Round-robin selection failed")

    async def disable_key(self, key: str, reset_time_ms: Optional[int] = None) -> None:
        """
        Disable a key until reset time or for the configured cooldown period.

        Args:
            key: The API key to disable
            reset_time_ms: Optional reset time in milliseconds since epoch. If provided,
                          the key will be disabled until this time. Otherwise, the default
                          cooldown period will be used.
        """
        async with self.lock:
            now = datetime.now()
            disabled_until = self._calculate_disable_time(now, reset_time_ms)

            self.disabled_until[key] = disabled_until
            logger.warning(
                "API key %s has been disabled until %s.",
                mask_key(key),
                disabled_until
            )

    def _calculate_disable_time(
            self,
            now: datetime,
            reset_time_ms: Optional[int]
    ) -> datetime:
        """
        Calculate when to re-enable a disabled key.

        Args:
            now: Current datetime
            reset_time_ms: Optional reset time from server

        Returns:
            Datetime when key should be re-enabled
        """
        if reset_time_ms:
            try:
                # Convert milliseconds to seconds and create datetime
                reset_datetime = datetime.fromtimestamp(reset_time_ms / 1000)

                # Ensure reset time is in the future
                if reset_datetime > now:
                    logger.info("Using server-provided reset time: %s", str(reset_datetime))
                    return reset_datetime
                else:
                    # Fallback to default cooldown if reset time is in the past
                    logger.warning(
                        "Server-provided reset time is in the past, "
                        "using default cooldown of %s seconds",
                        self.cooldown_seconds
                    )
            except Exception as e:
                # Fallback to default cooldown on error
                logger.error(
                    "Error processing reset time %s, using default cooldown: %s",
                    reset_time_ms,
                    e
                )

        # Use default cooldown period
        logger.info(
            "No reset time provided, using default cooldown of %s seconds",
            self.cooldown_seconds
        )
        return now + timedelta(seconds=self.cooldown_seconds)


def check_global_limit(data: str) -> Optional[str]:
    """
    Check for a global rate limit error message from OpenRouter.

    Args:
        data: Response message to check

    Returns:
        Error message if global limit detected, None otherwise
    """
    if isinstance(data, str) and GLOBAL_LIMIT_PATTERN in data:
        logger.warning("Model %s is overloaded.", data.split(' ', 1)[0])
        return GLOBAL_LIMIT_ERROR
    return None


def check_google_error(data: str) -> Optional[str]:
    """
    Check for Google RESOURCE_EXHAUSTED error.

    Args:
        data: Response data to check

    Returns:
        Error message if Google error detected, None otherwise
    """
    if not data:
        return None

    try:
        parsed_data = json.loads(data)
        if parsed_data.get("error", {}).get("status") == "RESOURCE_EXHAUSTED":
            return GOOGLE_LIMIT_ERROR
    except Exception as e:
        logger.info("Json.loads error %s", e)

    return None


async def check_rate_limit(data: Union[str, bytes]) -> Tuple[bool, Optional[int]]:
    """
    Check for rate limit error in response data.

    Args:
        data: Response data to check

    Returns:
        Tuple of (has_rate_limit_error, reset_time_ms)
    """
    has_rate_limit_error = False
    reset_time_ms = None

    try:
        err = json.loads(data)
    except Exception as e:
        logger.warning('Json.loads error %s', e)
        return False, None

    if not isinstance(err, dict) or "error" not in err:
        return False, None

    code = err["error"].get("code", 0)

    # Try to get X-RateLimit-Reset from headers
    try:
        x_rate_limit = int(err["error"]["metadata"]["headers"]["X-RateLimit-Reset"])
    except (TypeError, KeyError):
        x_rate_limit = 0

        # Check for global or Google-specific rate limits
        if code == RATE_LIMIT_ERROR_CODE:
            raw_message = err["error"].get("metadata", {}).get("raw", "")
            issue = check_global_limit(raw_message) or check_google_error(raw_message)

            if issue:
                global_delay = config["openrouter"]["global_rate_delay"]
                if global_delay:
                    logger.info("%s, waiting %s seconds.", issue, global_delay)
                    await asyncio.sleep(global_delay)
                return False, None

    if x_rate_limit > 0:
        has_rate_limit_error = True
        reset_time_ms = x_rate_limit
    elif code == RATE_LIMIT_ERROR_CODE:
        has_rate_limit_error = True

    return has_rate_limit_error, reset_time_ms


async def verify_access_key(authorization: Optional[str] = Header(None)) -> bool:
    """
    Verify the local access key for authentication.

    Args:
        authorization: Authorization header value

    Returns:
        True if authentication is successful

    Raises:
        HTTPException: If authentication fails
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    scheme, _, token = authorization.partition(" ")

    if scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authentication scheme")

    if token != config["server"]["access_key"]:
        raise HTTPException(status_code=401, detail="Invalid access key")

    return True


@asynccontextmanager
async def lifespan(app_: FastAPI):
    """Manage the lifespan of the HTTP client."""
    client_kwargs = {"timeout": DEFAULT_HTTP_TIMEOUT}


    app_.state.http_client = httpx.AsyncClient(**client_kwargs)
    yield
    await app_.state.http_client.aclose()


async def get_async_client(request: Request) -> httpx.AsyncClient:
    """Get the shared httpx client instance from the request."""
    return request.app.state.http_client


async def check_httpx_err(body: Union[str, bytes], api_key: Optional[str]) -> None:
    """
    Check the response body for a rate limit error and disable the key if found.

    Args:
        body: Response body to check
        api_key: The API key used for the request
    """
    if not api_key:
        return

    body_len = len(body)
    if body_len < MIN_BODY_SIZE_FOR_CHECK or body_len > MAX_BODY_SIZE_FOR_CHECK:
        return

    has_rate_limit_error, reset_time_ms = await check_rate_limit(body)

    if has_rate_limit_error:
        await key_manager.disable_key(api_key, reset_time_ms)


def remove_paid_models(body: bytes) -> bytes:
    """
    Filter out non-free models from the /models endpoint response.

    Args:
        body: Response body containing model data

    Returns:
        Filtered response body with only free models
    """
    pricing_keys = ['prompt', 'completion', 'request', 'image', 'web_search', 'internal_reasoning']

    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Error deserializing models to filter paid ones: %s", str(e))
        return body

    if not isinstance(data.get("data"), list):
        return body

    clear_data = [
        model for model in data["data"]
        if all(model.get("pricing", {}).get(k, "1") == "0" for k in pricing_keys)
    ]

    if clear_data:
        data["data"] = clear_data
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")

    return body


def prepare_forward_headers(request: Request) -> Dict[str, str]:
    """
    Prepare headers for forwarding, removing sensitive or connection-specific ones.

    Args:
        request: The incoming FastAPI request

    Returns:
        Dictionary of headers safe to forward
    """
    excluded_headers = {"host", "content-length", "connection", "authorization"}

    return {
        k: v for k, v in request.headers.items()
        if k.lower() not in excluded_headers
    }


def build_request_kwargs(
        request: Request,
        path: str,
        api_key: str
) -> Dict[str, Any]:
    """
    Build kwargs dictionary for httpx request.

    Args:
        request: The incoming FastAPI request
        path: API path to request
        api_key: API key to use (empty string for public endpoints)

    Returns:
        Dictionary of kwargs for httpx request
    """
    headers = prepare_forward_headers(request)

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return {
        "method": request.method,
        "url": f"{config['openrouter']['base_url']}{path}",
        "headers": headers,
        "params": request.query_params,
    }


async def handle_streaming_response(
        response: httpx.Response,
        path: str,
        api_key: str
) -> StreamingResponse:
    """
    Handle streaming (SSE) response from OpenRouter.

    Args:
        response: The httpx response to stream
        path: API path being requested
        api_key: API key used for the request

    Returns:
        FastAPI StreamingResponse
    """
    headers = dict(response.headers)
    headers.pop("content-encoding", None)
    headers.pop("Content-Encoding", None)

    async def sse_stream():
        last_json = ""
        try:
            async for line in response.aiter_lines():
                if line.startswith("data: {"):
                    last_json = line[6:]
                yield f"{line}\n\n".encode("utf-8")
        finally:
            await response.aclose()
            logger.info("Completed streaming response for path '%s'.", path)
            await check_httpx_err(last_json, api_key)

    return StreamingResponse(
        sse_stream(),
        status_code=response.status_code,
        media_type="text/event-stream",
        headers=headers
    )


async def handle_non_streaming_response(
        response: httpx.Response,
        path: str,
        api_key: str,
        attempt: int
) -> Response:
    """
    Handle non-streaming response from OpenRouter.

    Args:
        response: The httpx response
        path: API path being requested
        api_key: API key used for the request
        attempt: Current attempt number

    Returns:
        FastAPI Response
    """
    body = response.content
    headers = dict(response.headers)
    headers.pop("content-encoding", None)
    headers.pop("Content-Encoding", None)

    logger.info(
        "Success (Attempt %d/%d): Proxied non-streaming request for path '%s'. "
        "Key: %s. Status: %d.",
        attempt + 1,
        MAX_RETRY_ATTEMPTS,
        path,
        mask_key(api_key),
        response.status_code
    )

    return Response(
        content=body,
        status_code=response.status_code,
        media_type="application/json",
        headers=headers
    )


async def handle_http_error(
        error: httpx.HTTPStatusError,
        path: str,
        api_key: str,
        attempt: int
) -> None:
    """
    Handle HTTP errors from OpenRouter requests.

    Args:
        error: The HTTP error
        path: API path being requested
        api_key: API key used for the request
        attempt: Current attempt number

    Raises:
        HTTPException: Always raises after logging
    """
    try:
        error_content = await error.response.aread()
    except httpx.ResponseNotRead:
        error_content = b"<Streaming response content could not be read>"
    finally:
        await error.response.aclose()

    logger.error(
        "Failure (Attempt %d/%d): Request for path '%s' failed. "
        "Key: %s. Status: %d. Response: %s",
        attempt + 1,
        MAX_RETRY_ATTEMPTS,
        path,
        mask_key(api_key),
        error.response.status_code,
        error_content.decode(errors='ignore')
    )

    if error.response.status_code == RATE_LIMIT_ERROR_CODE:
        logger.error(
            "All %d attempts failed with status 429. Final key was %s.",
            MAX_RETRY_ATTEMPTS,
            mask_key(api_key)
        )

    await check_httpx_err(error.response.content, api_key)
    raise HTTPException(error.response.status_code, error.response.content) from error


# --- API Endpoints ---

# Create router
router = APIRouter()

# Initialize key manager
key_manager = KeyManager(
    keys=config["openrouter"]["keys"],
    cooldown_seconds=config["openrouter"]["rate_limit_cooldown"],
    strategy=config["openrouter"]["key_selection_strategy"],
    opts=config["openrouter"]["key_selection_opts"],
)


@router.api_route("/api/v1{path:path}", methods=["GET", "POST"])
async def proxy_endpoint(
        request: Request,
        path: str,
        authorization: Optional[str] = Header(None)
):
    """Main proxy endpoint for handling all requests to OpenRouter API."""
    is_public = any(
        f"/api/v1{path}".startswith(ep)
        for ep in config["openrouter"]["public_endpoints"]
    )

    if not is_public:
        await verify_access_key(authorization=authorization)

    full_url = str(request.url).replace(str(request.base_url), "/")
    api_key = "" if is_public else await key_manager.get_next_key()

    logger.info(
        "Proxying request to %s (Public: %s, key: %s)",
        full_url,
        is_public,
        mask_key(api_key))

    is_stream = False
    n_value = 1
    request_body = {}

    if request.method == "POST":
        try:
            body_bytes = await request.body()
            if body_bytes:
                request_body = json.loads(body_bytes)
                is_stream = request_body.get("stream", False)
                n_value = request_body.get("n", 1)

                if is_stream:
                    logger.info("Detected streaming request for path '%s'", path)
                if n_value > 1:
                    logger.info(
                        "Detected parallel request for path '%s' with n=%d",
                        path,
                        n_value
                    )
                if model := request_body.get("model"):
                    logger.info("Using model: %s", model)
        except Exception as e:
            logger.debug("Could not parse request body for path '%s': %s", path, str(e))

    # Handle parallel requests (n > 1)
    if n_value > 1:
        if is_stream:
            raise HTTPException(
                status_code=400,
                detail="Parameter 'n' is not supported in streaming mode."
            )
        return await handle_parallel_requests(request, path, api_key, request_body, n_value)

    # Handle single request
    return await proxy_with_httpx(request, path, api_key, is_stream)


async def proxy_with_httpx(
    request: Request,
    path: str,
    api_key: str,
    is_stream: bool
) -> Response:
    """
    Core logic to proxy a single request with detailed logging and retries.

    Args:
        request: The incoming FastAPI request
        path: API path to request
        api_key: API key to use
        is_stream: Whether this is a streaming request

    Returns:
        FastAPI Response (either Response or StreamingResponse)

    Raises:
        HTTPException: On errors after all retries exhausted
    """
    req_kwargs = build_request_kwargs(request, path, api_key)
    req_kwargs["content"] = await request.body()

    client = await get_async_client(request)

    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:

            openrouter_req = client.build_request(**req_kwargs)
            openrouter_resp = await client.send(openrouter_req, stream=is_stream)

            # Handle rate limit and server errors with retry
            if openrouter_resp.status_code in [RATE_LIMIT_ERROR_CODE, SERVER_ERROR_CODE]:
                response_body = await openrouter_resp.aread()
                await openrouter_resp.aclose()
                await check_httpx_err(response_body, api_key)

                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    logger.warning(
                        "Attempt %d/%d failed with status %d for path '%s' (key: %s). Retrying...",
                        attempt + 1,
                        MAX_RETRY_ATTEMPTS,
                        openrouter_resp.status_code,
                        path,
                        mask_key(api_key)
                    )
                    api_key = await key_manager.get_next_key()
                    req_kwargs = build_request_kwargs(request, path, api_key)
                    req_kwargs["content"] = await request.body()
                    continue

            openrouter_resp.raise_for_status()

            # Handle streaming response
            if is_stream:
                logger.info(
                    "Success (Attempt %d/%d): Initiating streaming response for path '%s'. "
                    "Key: %s. Status: %d.",
                    attempt + 1,
                    MAX_RETRY_ATTEMPTS,
                    path,
                    mask_key(api_key),
                    openrouter_resp.status_code
                )
                return await handle_streaming_response(openrouter_resp, path, api_key)

            # Handle non-streaming response
            return await handle_non_streaming_response(
                openrouter_resp,
                path,
                api_key,
                attempt
            )

        except httpx.HTTPStatusError as e:
            await handle_http_error(e, path, api_key, attempt)

        except httpx.ConnectError as e:
            logger.error(
                "Failure (Attempt %d/%d): Connection error for path '%s'. Key: %s. Error: %s",
                attempt + 1,
                MAX_RETRY_ATTEMPTS,
                path,
                mask_key(api_key),
                str(e)
            )
            raise HTTPException(503, "Unable to connect to OpenRouter API") from e

        except httpx.TimeoutException as e:
            logger.error(
                "Failure (Attempt %d/%d): Timeout for path '%s'. Key: %s. Error: %s",
                attempt + 1,
                MAX_RETRY_ATTEMPTS,
                path,
                mask_key(api_key),
                str(e)
            )
            raise HTTPException(504, "OpenRouter API request timed out") from e

        except Exception as e:
            logger.error(
                "Failure (Attempt %d/%d): Internal error for path '%s'. Key: %s. Error: %s",
                attempt + 1,
                MAX_RETRY_ATTEMPTS,
                path,
                mask_key(api_key),
                str(e),
                exc_info=True
            )
            raise HTTPException(status_code=500, detail="Internal Proxy Error") from e

    raise HTTPException(status_code=500, detail="Exhausted all retry attempts.")


async def handle_parallel_requests(
    request: Request,
    path: str,
    initial_api_key: str,
    request_body: Dict[str, Any],
    n: int
) -> JSONResponse:
    """
    Handle n > 1 by making parallel asynchronous requests.

    Args:
        request: The incoming FastAPI request
        path: API path to request
        initial_api_key: Initial API key (unused, keys are selected per request)
        request_body: Parsed request body
        n: Number of parallel completions to generate

    Returns:
        JSONResponse with combined results

    Raises:
        HTTPException: If all parallel requests fail
    """
    # Remove 'n' from body since each individual request will be n=1
    request_body.pop('n', None)

    # Create n asynchronous tasks
    tasks = [
        execute_single_request(request, path, request_body)
        for _ in range(n)
    ]

    # Run them in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("Parallel request %d/%d failed: %s", i + 1, n, result)
            continue

        # Successful result is a response dictionary
        response_choices = result.get("choices", [])
        if response_choices:
            choice = response_choices[0]
            choice["index"] = i  # Reassign index
            choices.append(choice)

        usage = result.get("usage", {})
        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_completion_tokens += usage.get("completion_tokens", 0)

    if not choices:
        raise HTTPException(
            status_code=502,
            detail="All parallel upstream requests failed."
        )

    # Assemble final response
    final_response = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request_body.get("model", "unknown-model"),
        "choices": choices,
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        }
    }

    return JSONResponse(content=final_response)


async def execute_single_request(
    request: Request,
    path: str,
    body: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute a single, non-streaming request and return the JSON response.

    Args:
        request: The incoming FastAPI request
        path: API path to request
        body: Request body dictionary

    Returns:
        JSON response dictionary

    Raises:
        Exception: On request failure
    """
    api_key = await key_manager.get_next_key()

    req_kwargs = {
        "method": "POST",
        "url": f"{config['openrouter']['base_url']}{path}",
        "headers": prepare_forward_headers(request),
        "json": body,
        "params": request.query_params,
    }

    if api_key:
        req_kwargs["headers"]["Authorization"] = f"Bearer {api_key}"

    client = await get_async_client(request)

    try:
        response = await client.request(**req_kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == RATE_LIMIT_ERROR_CODE:
            await key_manager.disable_key(api_key)
        raise e
    except Exception as e:
        raise e


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


# Create FastAPI app
app = FastAPI(
    title="OpenRouter API Proxy",
    description="Proxies requests to OpenRouter API and rotates API keys to bypass rate limits",
    version="1.0.0",
    lifespan=lifespan,
)

# Include routes
app.include_router(router)

# Entry point
if __name__ == "__main__":
    host = config["server"]["host"]
    port = config["server"]["port"]

    # Configure log level for HTTP access logs
    log_config = uvicorn.config.LOGGING_CONFIG
    http_log_level = config["server"].get("http_log_level", "INFO").upper()
    log_config["loggers"]["uvicorn.access"]["level"] = http_log_level
    logger.info("HTTP access log level set to %s", http_log_level)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=log_config,
        timeout_graceful_shutdown=60
    )