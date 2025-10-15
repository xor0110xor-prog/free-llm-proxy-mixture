# fastest_provider_proxy.py

# FastAPI proxy that races multiple downstream LLM providers and returns the fastest response.
# Extracted from the MOA aggregator for standalone fastest-response functionality.
# Features: Random provider selection, temporary banning for slow/failing providers.

import asyncio
import time
import uuid
import random
import json
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, AsyncGenerator, List, Optional
from pathlib import Path
from enum import Enum

import httpx
import yaml
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger


# --- Load Configuration from YAML ---
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {config_path}")
    return config


# Load global configuration
CONFIG = load_config()

# Extract configuration sections
TARGET_APIS = CONFIG['downstream_apis']
TIMEOUT_CONFIG = CONFIG['timeouts']
RETRY_CONFIG = CONFIG['retry']
SERVER_CONFIG = CONFIG['server']
RACING_CONFIG = CONFIG.get('racing', {})

# API Key for authentication
API_KEY: Optional[str] = CONFIG.get("api_key")

# --- Extracted Configuration Variables ---
HTTP_CLIENT_TIMEOUT = TIMEOUT_CONFIG['http_client_timeout']
MAX_RETRIES = RETRY_CONFIG['max_retries']
BASE_DELAY = RETRY_CONFIG['base_delay']
MAX_DELAY = RETRY_CONFIG['max_delay']
JITTER_RANGE = RETRY_CONFIG['jitter_range']

# Racing specific config
DEFAULT_NUM_RACERS = RACING_CONFIG.get('default_num_racers', 5)
MAX_NUM_RACERS = RACING_CONFIG.get('max_num_racers', 10)
INITIAL_BACKOFF_SECONDS = RACING_CONFIG.get('initial_backoff_seconds', 30)
SLOWNESS_THRESHOLD_MULTIPLIER = RACING_CONFIG.get('slowness_threshold_multiplier', 2.5)
STATE_FILE = RACING_CONFIG.get('state_file', 'fastest_provider_states.json')


# ===========================
# Provider State Management
# ===========================

class ProviderState(Enum):
    """States a provider can be in."""
    AVAILABLE = "available"
    BANNED = "banned"


class ProviderInfo:
    """Tracks a provider's performance and ban state."""

    def __init__(self, provider_id: str):
        self.provider_id = provider_id
        self.status: ProviderState = ProviderState.AVAILABLE
        self.failure_count: int = 0
        self.slow_count: int = 0
        self.available_at: float = 0.0
        self.last_response_time: float = 0.0
        self.avg_response_time: float = 0.0
        self.total_requests: int = 0
        self.successful_requests: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize provider info to dictionary."""
        return {
            "provider_id": self.provider_id,
            "status": self.status.value,
            "failure_count": self.failure_count,
            "slow_count": self.slow_count,
            "available_at": self.available_at,
            "last_response_time": self.last_response_time,
            "avg_response_time": self.avg_response_time,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderInfo":
        """Deserialize provider info from dictionary."""
        provider = cls(data["provider_id"])
        provider.status = ProviderState(data.get("status", "available"))
        provider.failure_count = data.get("failure_count", 0)
        provider.slow_count = data.get("slow_count", 0)
        provider.available_at = data.get("available_at", 0.0)
        provider.last_response_time = data.get("last_response_time", 0.0)
        provider.avg_response_time = data.get("avg_response_time", 0.0)
        provider.total_requests = data.get("total_requests", 0)
        provider.successful_requests = data.get("successful_requests", 0)
        return provider


class ProviderManager:
    """Manages provider performance tracking and temporary banning."""

    def __init__(self, initial_backoff: int = INITIAL_BACKOFF_SECONDS):
        self.providers: Dict[str, ProviderInfo] = {}
        self.initial_backoff = initial_backoff
        self._lock = asyncio.Lock()

        # Initialize providers from config
        for provider_id in TARGET_APIS.keys():
            self.providers[provider_id] = ProviderInfo(provider_id)

        self._load_state()

    def _load_state(self):
        """Load provider states from disk."""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r') as f:
                    states = json.load(f)
                for provider_id, data in states.items():
                    if provider_id in self.providers:
                        self.providers[provider_id] = ProviderInfo.from_dict(data)
                logger.info(f"Loaded {len(self.providers)} provider states from {STATE_FILE}")
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load provider states: {e}")

    async def _save_state(self):
        """Save provider states to disk."""
        try:
            states = {pid: prov.to_dict() for pid, prov in self.providers.items()}
            with open(STATE_FILE, 'w') as f:
                json.dump(states, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save provider states: {e}")

    async def get_available_providers(self) -> List[str]:
        """Get list of currently available (non-banned) provider IDs."""
        async with self._lock:
            now = time.time()
            available = []

            for provider_id, provider in self.providers.items():
                # Check if ban has expired
                if provider.status == ProviderState.BANNED:
                    if now >= provider.available_at:
                        provider.status = ProviderState.AVAILABLE
                        provider.failure_count = 0
                        provider.slow_count = 0
                        logger.info(f"Provider {provider_id} is now available after cooldown")

                if provider.status == ProviderState.AVAILABLE:
                    available.append(provider_id)

            return available

    async def report_success(self, provider_id: str, response_time: float,
                            winner_time: Optional[float] = None):
        """Report successful response from a provider."""
        async with self._lock:
            if provider_id not in self.providers:
                return

            provider = self.providers[provider_id]
            provider.total_requests += 1
            provider.successful_requests += 1
            provider.last_response_time = response_time

            # Update average response time
            if provider.avg_response_time == 0:
                provider.avg_response_time = response_time
            else:
                provider.avg_response_time = (provider.avg_response_time * 0.8 + response_time * 0.2)

            # Check if this provider was significantly slower than the winner
            if winner_time and winner_time < response_time:
                slowness_ratio = response_time / winner_time
                if slowness_ratio >= SLOWNESS_THRESHOLD_MULTIPLIER:
                    provider.slow_count += 1
                    logger.warning(
                        f"Provider {provider_id} was {slowness_ratio:.1f}x slower than winner "
                        f"({response_time:.2f}s vs {winner_time:.2f}s). Slow count: {provider.slow_count}"
                    )

                    # Ban if too many slow responses
                    if provider.slow_count >= 3:
                        await self._ban_provider(provider, "slowness")

            await self._save_state()

    async def report_failure(self, provider_id: str):
        """Report failure from a provider and apply temporary ban if needed."""
        async with self._lock:
            if provider_id not in self.providers:
                return

            provider = self.providers[provider_id]
            provider.total_requests += 1
            provider.failure_count += 1

            logger.warning(
                f"Provider {provider_id} failed. Failure count: {provider.failure_count}"
            )

            # Ban after failures
            if provider.failure_count >= 2:
                await self._ban_provider(provider, "failures")

            await self._save_state()

    async def _ban_provider(self, provider: ProviderInfo, reason: str):
        """Ban a provider with exponential backoff."""
        provider.status = ProviderState.BANNED

        # Calculate exponential backoff based on total penalties
        total_penalties = provider.failure_count + provider.slow_count
        backoff_duration = self.initial_backoff * (2 ** (total_penalties - 1))
        # Cap at 1 hour
        backoff_duration = min(backoff_duration, 3600)

        provider.available_at = time.time() + backoff_duration

        logger.warning(
            f"Provider {provider.provider_id} BANNED for {backoff_duration}s due to {reason}. "
            f"Failures: {provider.failure_count}, Slow: {provider.slow_count}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all providers."""
        stats = {}
        for provider_id, provider in self.providers.items():
            success_rate = (provider.successful_requests / provider.total_requests * 100
                          if provider.total_requests > 0 else 0)
            stats[provider_id] = {
                "status": provider.status.value,
                "total_requests": provider.total_requests,
                "successful_requests": provider.successful_requests,
                "success_rate": f"{success_rate:.1f}%",
                "avg_response_time": f"{provider.avg_response_time:.2f}s",
                "failure_count": provider.failure_count,
                "slow_count": provider.slow_count,
            }
            if provider.status == ProviderState.BANNED:
                remaining = max(0, provider.available_at - time.time())
                stats[provider_id]["banned_for_seconds"] = int(remaining)
        return stats



# --- FastAPI Application Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages the lifecycle of the shared HTTP client and provider manager."""
    app.state.http_client = httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT)
    app.state.provider_manager = ProviderManager()
    logger.info(f"Initialized ProviderManager with {len(app.state.provider_manager.providers)} providers")
    yield
    await app.state.http_client.aclose()


app = FastAPI(
    title="Fastest Provider Racing Proxy",
    description="Races multiple downstream LLM APIs and returns the fastest successful response.",
    lifespan=lifespan
)


# --- Authentication Functions ---
async def verify_api_key(authorization: Optional[str] = Header(None)) -> None:
    """Verify API key from authorization header."""
    if API_KEY and (
        not authorization or
        not authorization.startswith("Bearer ") or
        authorization[7:] != API_KEY
    ):
        raise HTTPException(status_code=401, detail="Invalid API key")


# --- Core Logic: API Calling and Retries ---
async def exponential_backoff_delay(attempt: int, base_delay: float = BASE_DELAY,
                                    max_delay: float = MAX_DELAY) -> float:
    """Calculate exponential backoff delay with jitter."""
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = delay * JITTER_RANGE * (2 * random.random() - 1)
    return max(0, delay + jitter)


def is_retryable_error(error: Exception) -> bool:
    """Determine if an error is retryable based on its type and HTTP status code."""
    if isinstance(error, (httpx.TimeoutException, httpx.RequestError)):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        code = error.response.status_code
        return code in (429, 502, 503, 504) or 500 <= code < 600
    # JSON parsing failures (partial/garbled response)
    if isinstance(error, (ValueError, json.JSONDecodeError)):
        return True
    return False


async def call_downstream_api_with_retry(
        client: httpx.AsyncClient,
        target_config: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
        stream: bool = False,
        max_retries: int = MAX_RETRIES
) -> Dict[str, Any]:
    """
    Asynchronously calls a single downstream API with robust retry logic and tool support.
    Returns either a dict (for non-streaming) or httpx.Response (for streaming).
    """
    url = target_config["base_url"].rstrip('/') + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json" if not stream else "text/event-stream",
        "Authorization": f"Bearer {target_config['api_key']}"
    }

    payload = {
        "model": target_config["model"],
        "messages": messages,
        "stream": stream,
    }

    # Add tool-related parameters if provided
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = parallel_tool_calls

    # Add passthrough OpenAI parameters
    passthrough = target_config.get("passthrough", {})
    payload.update(passthrough)

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                delay = await exponential_backoff_delay(attempt - 1)
                await asyncio.sleep(delay)

            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()

            # For streaming, return the response object directly
            if stream:
                return {"stream_response": response}

            # Explicitly handle JSON parsing for non-streaming
            try:
                return response.json()
            except (ValueError, json.JSONDecodeError) as e:
                last_error = e
                response_text = response.text if hasattr(response, 'text') else 'N/A'
                logger.error(
                    f"JSON decode error from {url}:\n"
                    f"  Status: {response.status_code}\n"
                    f"  Headers: {dict(response.headers)}\n"
                    f"  Full Response Body:\n{response_text}\n"
                    f"  Error: {str(e)}"
                )
                if attempt < max_retries:
                    logger.warning(f"Retrying JSON decode (attempt {attempt + 1}/{max_retries + 1})")
                    continue
                raise

        except httpx.HTTPStatusError as error:
            last_error = error
            response_text = ""
            try:
                response_text = error.response.text
            except Exception:
                response_text = "Unable to read response body"

            logger.error(
                f"HTTP error from {url}:\n"
                f"  Status: {error.response.status_code}\n"
                f"  Reason: {error.response.reason_phrase if hasattr(error.response, 'reason_phrase') else 'N/A'}\n"
                f"  Headers: {dict(error.response.headers)}\n"
                f"  Full Response Body:\n{response_text}\n"
                f"  Request Payload:\n{json.dumps(payload, indent=2)}"
            )

            if attempt < max_retries and is_retryable_error(error):
                logger.warning(f"Retrying after HTTP error (attempt {attempt + 1}/{max_retries + 1})")
            else:
                break

        except httpx.RequestError as error:
            last_error = error
            logger.error(
                f"Request error calling {url}:\n"
                f"  Error type: {type(error).__name__}\n"
                f"  Error message: {str(error)}\n"
                f"  Request URL: {url}\n"
                f"  Request Payload:\n{json.dumps(payload, indent=2)}"
            )

            if attempt < max_retries and is_retryable_error(error):
                logger.warning(f"Retrying after request error (attempt {attempt + 1}/{max_retries + 1})")
            else:
                break

        except Exception as error:
            last_error = error
            logger.error(
                f"Unexpected error calling {url}:\n"
                f"  Error type: {type(error).__name__}\n"
                f"  Error message: {str(error)}\n"
                f"  Request Payload:\n{json.dumps(payload, indent=2)}",
                exc_info=True
            )

            if attempt < max_retries and is_retryable_error(error):
                logger.warning(f"Retrying after unexpected error (attempt {attempt + 1}/{max_retries + 1})")
            else:
                break

    # Prepare detailed error response
    error_details = "Unknown error"
    if isinstance(last_error, httpx.HTTPStatusError):
        try:
            # Include full response text without truncation
            error_details = f"HTTP {last_error.response.status_code}: {last_error.response.text}"
        except Exception:
            error_details = f"HTTP {last_error.response.status_code}: Unable to read response"
    elif isinstance(last_error, httpx.RequestError):
        error_details = f"{type(last_error).__name__}: {str(last_error)}"
    else:
        error_details = f"{type(last_error).__name__}: {str(last_error)}"

    return {
        "error": f"API call failed after {attempt + 1} attempts",
        "details": error_details,
        "retry_attempts": attempt,
        "url": url
    }


# --- Racing Logic ---
async def call_fastest_api_streaming(
        request: Request,
        num_apis: int = DEFAULT_NUM_RACERS
) -> StreamingResponse:
    """
    Races multiple downstream APIs in streaming mode and returns the fastest stream.

    Args:
        request: FastAPI request object
        num_apis: Number of APIs to race
    """
    request_id = f"fastest-stream-{uuid.uuid4()}"
    start_time = time.time()

    try:
        # Use cached JSON data from endpoint
        data = request.state.cached_json
        messages = data.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="Parameter 'messages' is required.")

        # Extract tool-calling parameters
        tools = data.get("tools")
        tool_choice = data.get("tool_choice")
        parallel_tool_calls = data.get("parallel_tool_calls")

        # Extract OpenAI passthrough parameters
        passthrough = {k: v for k, v in data.items() if k in (
            "temperature", "max_tokens", "top_p", "stop", "presence_penalty",
            "frequency_penalty", "response_format", "logit_bias", "seed", "n", "user"
        )}

        # Allow client to override racer count
        requested = int(data.get("num_racers", num_apis))
        requested = max(1, min(requested, MAX_NUM_RACERS))

        client: httpx.AsyncClient = request.app.state.http_client
        provider_manager: ProviderManager = request.app.state.provider_manager

        # Get available (non-banned) providers
        available_provider_ids = await provider_manager.get_available_providers()
        if not available_provider_ids:
            raise HTTPException(
                status_code=503,
                detail="All providers are currently banned. Please try again later."
            )

        # Select random APIs to race from available providers
        num_apis = min(requested, len(available_provider_ids))
        if requested > len(available_provider_ids):
            logger.warning(f"[{request_id}] Requested {requested} APIs, but only {len(available_provider_ids)} available")

        selected_provider_ids = random.sample(available_provider_ids, num_apis)
        selected_apis = [(pid, {**TARGET_APIS[pid], "passthrough": passthrough}) for pid in selected_provider_ids]
        logger.info(f"[{request_id}] Racing {num_apis} streaming APIs: {selected_provider_ids}")

        # Create racing tasks
        tasks = []
        task_metadata = []
        task_start_times = {}

        for api_id, api_config in selected_apis:
            task = asyncio.create_task(
                call_downstream_api_with_retry(
                    client, api_config, messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    parallel_tool_calls=parallel_tool_calls,
                    stream=True
                )
            )
            tasks.append(task)
            task_metadata.append({
                "api_id": api_id,
                "model": api_config.get("model", "unknown"),
                "task": task
            })
            task_start_times[api_id] = time.time()

        winner_response = None
        winner_metadata = None

        # Wait for first successful stream connection
        pending_tasks = set(tasks)
        while pending_tasks and winner_response is None:
            done, pending_tasks = await asyncio.wait(
                pending_tasks,
                return_when=asyncio.FIRST_COMPLETED
            )

            for completed_task in done:
                # Find metadata for this task
                task_meta = None
                for meta in task_metadata:
                    if meta["task"] == completed_task:
                        task_meta = meta
                        break

                if not task_meta:
                    continue

                api_id = task_meta["api_id"]

                try:
                    result = await completed_task

                    # Skip if error
                    if "error" in result:
                        logger.error(
                            f"[{request_id}] Provider {api_id} FAILED in streaming race:\n"
                            f"  Error: {result.get('error', 'Unknown error')}\n"
                            f"  Details: {result.get('details', 'No details')}\n"
                            f"  URL: {result.get('url', 'N/A')}\n"
                            f"  Retry attempts: {result.get('retry_attempts', 0)}"
                        )
                        # Report failure immediately
                        await provider_manager.report_failure(api_id)
                        continue

                    # Check if we got a stream response
                    if "stream_response" in result:
                        winner_metadata = task_meta
                        winner_response = result["stream_response"]
                        break

                except Exception as e:
                    logger.error(
                        f"[{request_id}] Provider {api_id} EXCEPTION in streaming race:\n"
                        f"  Exception type: {type(e).__name__}\n"
                        f"  Exception message: {str(e)}",
                        exc_info=True
                    )
                    # Report failure
                    await provider_manager.report_failure(api_id)
                    continue

        # Cancel remaining tasks immediately
        if pending_tasks:
            logger.info(f"[{request_id}] Cancelling {len(pending_tasks)} remaining streaming tasks")
            for task in pending_tasks:
                meta = next((m for m in task_metadata if m["task"] == task), None)
                if meta:
                    await provider_manager.report_failure(meta["api_id"])
                task.cancel()
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        if winner_response is None:
            raise HTTPException(
                status_code=502,
                detail="All APIs failed in streaming racing mode"
            )

        winner_api_id = winner_metadata["api_id"]
        time_to_first_chunk = time.time() - task_start_times[winner_api_id]

        logger.info(
            f"[{request_id}] Streaming winner: {winner_metadata['api_id']} "
            f"(first chunk in {time_to_first_chunk:.2f}s)"
        )

        # Create streaming generator
        async def stream_generator():
            """Proxy the winning stream and track performance."""
            total_chunks = 0
            try:
                async for chunk in winner_response.aiter_bytes():
                    total_chunks += 1
                    yield chunk

                # Report success after stream completes
                elapsed = time.time() - start_time
                await provider_manager.report_success(winner_api_id, elapsed)
                logger.info(
                    f"[{request_id}] Stream completed: {winner_api_id} "
                    f"({total_chunks} chunks in {elapsed:.2f}s)"
                )
            except Exception as e:
                logger.error(f"[{request_id}] Error streaming from {winner_api_id}: {e}")
                await provider_manager.report_failure(winner_api_id)
                raise

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Winner-API": winner_metadata["api_id"],
                "X-Winner-Model": winner_metadata["model"],
                "X-Request-ID": request_id
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error in fastest streaming racing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Streaming racing error: {e}")


async def call_fastest_api(
        request: Request,
        num_apis: int = DEFAULT_NUM_RACERS
) -> JSONResponse:
    """
    Races multiple downstream APIs and returns the fastest successful response.

    Args:
        request: FastAPI request object
        num_apis: Number of APIs to race
    """
    request_id = f"fastest-{uuid.uuid4()}"
    start_time = time.time()

    try:
        # Use cached JSON data from endpoint
        data = request.state.cached_json
        messages = data.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="Parameter 'messages' is required.")

        # Extract tool-calling parameters
        tools = data.get("tools")
        tool_choice = data.get("tool_choice")
        parallel_tool_calls = data.get("parallel_tool_calls")

        # Extract OpenAI passthrough parameters
        passthrough = {k: v for k, v in data.items() if k in (
            "temperature", "max_tokens", "top_p", "stop", "presence_penalty",
            "frequency_penalty", "response_format", "logit_bias", "seed", "n", "user"
        )}

        # Allow client to override racer count
        requested = int(data.get("num_racers", num_apis))
        requested = max(1, min(requested, MAX_NUM_RACERS))

        client: httpx.AsyncClient = request.app.state.http_client
        provider_manager: ProviderManager = request.app.state.provider_manager

        # Get available (non-banned) providers
        available_provider_ids = await provider_manager.get_available_providers()
        if not available_provider_ids:
            raise HTTPException(
                status_code=503,
                detail="All providers are currently banned. Please try again later."
            )

        # Select random APIs to race from available providers
        num_apis = min(requested, len(available_provider_ids))
        if requested > len(available_provider_ids):
            logger.warning(f"[{request_id}] Requested {requested} APIs, but only {len(available_provider_ids)} available")

        selected_provider_ids = random.sample(available_provider_ids, num_apis)
        selected_apis = [(pid, {**TARGET_APIS[pid], "passthrough": passthrough}) for pid in selected_provider_ids]
        logger.info(f"[{request_id}] Racing {num_apis} APIs: {selected_provider_ids}")

        # Create racing tasks
        tasks = []
        task_metadata = []
        task_start_times = {}

        for api_id, api_config in selected_apis:
            task = asyncio.create_task(
                call_downstream_api_with_retry(
                    client, api_config, messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    parallel_tool_calls=parallel_tool_calls
                )
            )
            tasks.append(task)
            task_metadata.append({
                "api_id": api_id,
                "model": api_config.get("model", "unknown"),
                "task": task
            })
            task_start_times[api_id] = time.time()

        winner_result = None
        winner_metadata = None
        completed_results = {}  # Track all completed tasks for reporting

        # Wait for first successful completion
        pending_tasks = set(tasks)
        while pending_tasks and winner_result is None:
            done, pending_tasks = await asyncio.wait(
                pending_tasks,
                return_when=asyncio.FIRST_COMPLETED
            )

            for completed_task in done:
                # Find metadata for this task
                task_meta = None
                for meta in task_metadata:
                    if meta["task"] == completed_task:
                        task_meta = meta
                        break

                if not task_meta:
                    continue

                api_id = task_meta["api_id"]
                response_time = time.time() - task_start_times[api_id]

                try:
                    result = await completed_task

                    # Skip if error
                    if "error" in result:
                        logger.error(
                            f"[{request_id}] Provider {api_id} FAILED in race:\n"
                            f"  Error: {result.get('error', 'Unknown error')}\n"
                            f"  Details: {result.get('details', 'No details')}\n"
                            f"  URL: {result.get('url', 'N/A')}\n"
                            f"  Retry attempts: {result.get('retry_attempts', 0)}\n"
                            f"  Time elapsed: {response_time:.2f}s"
                        )
                        # Report failure immediately
                        await provider_manager.report_failure(api_id)
                        continue

                    # Found winner!
                    winner_metadata = task_meta
                    winner_result = result
                    completed_results[api_id] = response_time
                    break

                except Exception as e:
                    logger.error(
                        f"[{request_id}] Provider {api_id} EXCEPTION in race:\n"
                        f"  Exception type: {type(e).__name__}\n"
                        f"  Exception message: {str(e)}\n"
                        f"  Time elapsed: {response_time:.2f}s",
                        exc_info=True
                    )
                    # Report failure
                    await provider_manager.report_failure(api_id)
                    continue

        # Wait briefly for any other tasks that are almost done (to detect slowness)
        if pending_tasks:
            logger.info(f"[{request_id}] Waiting briefly for {len(pending_tasks)} remaining tasks to complete...")
            try:
                await asyncio.wait(pending_tasks, timeout=0.5)
            except asyncio.TimeoutError:
                pass

            # Check which tasks completed
            for task in list(pending_tasks):
                if task.done():
                    task_meta = None
                    for meta in task_metadata:
                        if meta["task"] == task:
                            task_meta = meta
                            break

                    if task_meta:
                        api_id = task_meta["api_id"]
                        response_time = time.time() - task_start_times[api_id]
                        try:
                            result = task.result()
                            if "error" not in result:
                                completed_results[api_id] = response_time
                        except Exception:
                            pass  # Already logged

            # Cancel any truly remaining tasks and penalize as slow
            remaining = [t for t in pending_tasks if not t.done()]
            if remaining:
                logger.info(f"[{request_id}] Cancelling {len(remaining)} remaining racing tasks (penalizing as slow)")
                # Penalize remaining tasks as failures (they didn't complete in reasonable time)
                for task in remaining:
                    meta = next((m for m in task_metadata if m["task"] == task), None)
                    if meta:
                        await provider_manager.report_failure(meta["api_id"])
                    task.cancel()
                await asyncio.gather(*remaining, return_exceptions=True)

        if winner_result is None:
            raise HTTPException(
                status_code=502,
                detail="All APIs failed in racing mode"
            )

        # Report success to provider manager
        winner_api_id = winner_metadata["api_id"]
        winner_time = completed_results[winner_api_id]

        # Report winner success
        await provider_manager.report_success(winner_api_id, winner_time)

        # Report other completed tasks with winner time for comparison
        for api_id, response_time in completed_results.items():
            if api_id != winner_api_id:
                await provider_manager.report_success(api_id, response_time, winner_time)

        elapsed_time = time.time() - start_time
        winner_result["id"] = request_id
        winner_result["model"] = f"fastest-{winner_metadata['api_id']}"

        if "metadata" not in winner_result:
            winner_result["metadata"] = {}

        winner_result["metadata"].update({
            "winner_api": winner_metadata["api_id"],
            "winner_model": winner_metadata["model"],
            "served_model": winner_metadata["model"],
            "elapsed_seconds": round(elapsed_time, 2),
            "mode": "racing",
            "num_racers": num_apis,
            "tested_apis": [meta["api_id"] for meta in task_metadata]
        })

        logger.info(
            f"[{request_id}] Racing winner: {winner_metadata['api_id']} "
            f"({winner_metadata['model']}) in {elapsed_time:.2f}s"
        )

        return JSONResponse(content=winner_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error in fastest API racing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Racing error: {e}")


# --- API Endpoints ---
@app.post("/v1/chat/completions")
async def fastest_racing_endpoint(request: Request, authorization: Optional[str] = Header(None)):
    """
    Main endpoint: Races all configured downstream APIs and returns the fastest response.
    Supports both streaming and non-streaming modes.
    """
    await verify_api_key(authorization)

    # Parse request body once
    try:
        body_bytes = await request.body()
        data = json.loads(body_bytes)
        stream = data.get("stream", False)

        # Create a new request object with the parsed data to avoid double-reading
        # Store the data in the request state for handler functions
        request.state.cached_json = data

        if stream:
            return await call_fastest_api_streaming(request, num_apis=DEFAULT_NUM_RACERS)
        else:
            return await call_fastest_api(request, num_apis=DEFAULT_NUM_RACERS)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None)) -> JSONResponse:
    """Lists available models (downstream APIs)."""
    await verify_api_key(authorization)
    models = []
    for api_id, config in TARGET_APIS.items():
        # Redact sensitive fields
        redacted = {k: v for k, v in config.items()
                   if k.lower() not in ("api_key", "authorization", "auth", "headers")}
        models.append({
            "id": api_id,
            "object": "model",
            "owned_by": "fastest-proxy",
            "model_info": redacted
        })
    return JSONResponse(content={"object": "list", "data": models})


@app.get("/v1/stats")
async def provider_stats(authorization: Optional[str] = Header(None)):
    """Get performance statistics for all providers."""
    await verify_api_key(authorization)
    manager: ProviderManager = app.state.provider_manager
    return JSONResponse(content=manager.get_stats())


@app.get("/health")
async def health_check(authorization: Optional[str] = Header(None)):
    """Provides a health check and configuration overview."""
    await verify_api_key(authorization)
    manager: ProviderManager = app.state.provider_manager
    available_providers = await manager.get_available_providers()

    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "config": {
            "default_num_racers": DEFAULT_NUM_RACERS,
            "max_num_racers": MAX_NUM_RACERS,
            "max_retries": MAX_RETRIES,
            "tool_calls_supported": True,
            "streaming_supported": True
        },
        "available_apis": list(TARGET_APIS.keys()),
        "total_providers": len(TARGET_APIS),
        "available_providers": available_providers,
        "banned_providers": len(TARGET_APIS) - len(available_providers)
    }


# --- Main Execution Block ---
if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting Fastest Provider Racing Proxy on port {SERVER_CONFIG['port']}...")
    logger.info(f"Default racers: {DEFAULT_NUM_RACERS}, Max racers: {MAX_NUM_RACERS}")
    logger.info(f"Available providers: {list(TARGET_APIS.keys())}")
    uvicorn.run(app, host=SERVER_CONFIG['host'], port=SERVER_CONFIG['port'])
