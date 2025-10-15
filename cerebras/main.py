import asyncio
import json
import os
import random
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union

import httpx
import yaml
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger

# ===========================
# Constants
# ===========================

class Constants:
    """Centralized constants for Cerebras configuration."""

    # API Configuration
    CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"
    RATE_LIMIT_COOLDOWN = 120  # Base cooldown in seconds
    STATE_FILE = "cerebras_api_key_state.json"

    # Multi-pass configuration
    CLEVER_PASS_COUNT = 3

    # Request configuration
    REQUEST_TIMEOUT = 300.0
    DEFAULT_TIMEOUT = 30.0
    MAX_KEEPALIVE_CONNECTIONS = 20
    MAX_CONNECTIONS = 100

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

    # API Configuration
    API_KEY: Optional[str] = config.get("api_key")

    # Cerebras Configuration
    CEREBRAS_KEYS: List[str] = config.get("cerebras", {}).get("keys", [])
    CEREBRAS_RATE_LIMIT_COOLDOWN: int = config.get("cerebras", {}).get("rate_limit_cooldown", Constants.RATE_LIMIT_COOLDOWN)
    CEREBRAS_STATE_FILE: str = config.get("cerebras", {}).get("state_file", Constants.STATE_FILE)

    # Model Pools
    ALL_MODELS: List[str] = config.get("cerebras", {}).get("all_models", [])
    CLEVER_MODELS: List[str] = config.get("cerebras", {}).get("clever_models", [])
    MERGING_MODELS: List[str] = config.get("cerebras", {}).get("merging_models", [])

    # Multi-pass settings
    CLEVER_PASS_COUNT: int = config.get("cerebras", {}).get("clever_pass_count", Constants.CLEVER_PASS_COUNT)




# ===========================
# API Key Management
# ===========================

class ApiKeyInfo:
    """Represents an API key with its state and rate limiting information."""

    def __init__(self, key: str):
        self.key = key
        self.rate_limited_until: Dict[str, float] = {}
        self.last_used: float = 0.0
        self.rate_limit_history: Dict[str, List[float]] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize key info to dictionary."""
        return {
            "key": self.key,
            "rate_limited_until": self.rate_limited_until,
            "last_used": self.last_used,
            "rate_limit_history": self.rate_limit_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApiKeyInfo":
        """Deserialize key info from dictionary."""
        key_info = cls(data["key"])
        key_info.rate_limited_until = data.get("rate_limited_until", {})
        key_info.last_used = data.get("last_used", 0.0)
        key_info.rate_limit_history = data.get("rate_limit_history", {})
        return key_info


class ApiKeyManager:
    """Manages API keys with rotation and rate limiting."""

    def __init__(self):
        self.api_keys: List[ApiKeyInfo] = []
        self._lock = threading.Lock()
        self._request_counter = 0
        self._counter_lock = threading.Lock()

    def load_api_keys(self):
        """Load API keys from configuration and restore state from file."""
        logger.info("Loading Cerebras API keys from configuration...")

        for key in Settings.CEREBRAS_KEYS:
            self.api_keys.append(ApiKeyInfo(key))
            logger.info(f"  - Loaded key ending in ...{key[-4:]}")

        if not self.api_keys:
            logger.error("--- FATAL ERROR --- No API keys found in configuration.")
            exit(1)

        # Load state from file
        self._load_state()
        logger.info(f"Successfully loaded {len(self.api_keys)} API key(s).")

    def _load_state(self):
        """Load rate limit state from file."""
        try:
            if os.path.exists(Settings.CEREBRAS_STATE_FILE):
                with open(Settings.CEREBRAS_STATE_FILE, "r") as f:
                    saved_state = json.load(f)

                logger.info(f"Found state file '{Settings.CEREBRAS_STATE_FILE}'. Restoring rate limit history...")

                for key_info in self.api_keys:
                    key_value = key_info.key
                    if key_value in saved_state:
                        state = saved_state[key_value]
                        key_info.rate_limited_until = state.get("rate_limited_until", {})
                        key_info.rate_limit_history = state.get("rate_limit_history", {})
                        logger.info(f"  - Restored state for key ending in ...{key_value[-4:]}")

        except FileNotFoundError:
            logger.info(f"State file '{Settings.CEREBRAS_STATE_FILE}' not found. Starting with fresh state.")
        except json.JSONDecodeError:
            logger.warning(f"Could not parse state file '{Settings.CEREBRAS_STATE_FILE}'. Starting with fresh state.")
        except Exception as e:
            logger.warning(f"Unexpected error loading state file: {e}. Starting with fresh state.")

    def _save_state(self):
        """Save current state to file."""
        try:
            state_to_save = {key_info.key: key_info.to_dict() for key_info in self.api_keys}
            with open(Settings.CEREBRAS_STATE_FILE, "w") as f:
                json.dump(state_to_save, f, indent=2)
        except IOError as e:
            logger.error(f"Could not write to state file {Settings.CEREBRAS_STATE_FILE}: {e}")

    def get_next_api_key(self, model: str) -> Optional[ApiKeyInfo]:
        """Get the next available API key for the specified model."""
        with self._lock:
            now = time.time()
            available_keys = [
                key_info for key_info in self.api_keys
                if now >= key_info.rate_limited_until.get(model, 0)
            ]

            if not available_keys:
                return None

            # Return least recently used key
            best_key = min(available_keys, key=lambda k: k.last_used)
            best_key.last_used = now

            logger.info(f"Selected API key ending in ...{best_key.key[-4:]} for model {model}")
            return best_key

    def mark_key_rate_limited(self, key_info: ApiKeyInfo, model: str):
        """Mark a key as rate limited for a specific model."""
        with self._lock:
            now = time.time()
            one_hour_ago = now - 3600

            # Update rate limit history
            history = key_info.rate_limit_history.get(model, [])
            recent_history = [t for t in history if t > one_hour_ago]
            recent_history.append(now)
            key_info.rate_limit_history[model] = recent_history

            # Calculate cooldown with exponential backoff
            current_cooldown = Settings.CEREBRAS_RATE_LIMIT_COOLDOWN * len(recent_history)
            cooldown_until = now + current_cooldown

            key_info.rate_limited_until[model] = cooldown_until
            self._save_state()

        message_suffix = f" (cooldown increased due to {len(recent_history)} hits in the last hour)" if len(recent_history) > 1 else ""
        logger.warning(
            f"Rate limit hit for key ...{key_info.key[-4:]} on model {model}. "
            f"Cooling down for {current_cooldown}s.{message_suffix}"
        )




# ===========================
# Cerebras API Client
# ===========================

class CerebrasApiClient:
    """Client for Cerebras API operations."""

    def __init__(self, api_key_manager: ApiKeyManager):
        self.api_key_manager = api_key_manager

    async def forward_request(
        self,
        request: Request,
        request_data: Dict[str, Any]
    ) -> httpx.Response:
        """Forward a request to Cerebras with key rotation and retry logic."""
        model = request_data["model"]
        http_client: httpx.AsyncClient = request.app.state.http_client

        # Try each available key
        for attempt in range(len(self.api_key_manager.api_keys)):
            api_key_info = self.api_key_manager.get_next_api_key(model)

            if api_key_info is None:
                logger.error(f"All {len(self.api_key_manager.api_keys)} keys are rate-limited for model {model}")
                raise HTTPException(
                    status_code=429,
                    detail=f"All available API keys are rate-limited for model '{model}'"
                )

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key_info.key}",
                "Accept": "application/json",
            }

            try:
                response = await http_client.post(
                    Constants.CEREBRAS_API_URL,
                    headers=headers,
                    json=request_data,
                    timeout=Constants.REQUEST_TIMEOUT
                )

                if response.status_code == 429:
                    logger.warning(f"Rate limit from Cerebras for model {model}")
                    self.api_key_manager.mark_key_rate_limited(api_key_info, model)
                    continue

                response.raise_for_status()
                return response

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error from Cerebras: {e.response.status_code} - {e.response.text}")
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=f"Upstream API error: {e.response.text}"
                )
            except httpx.RequestError as e:
                logger.error(f"Network error: {e}")
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to connect to Cerebras API: {e}"
                )

        raise HTTPException(
            status_code=429,
            detail="All API keys are rate-limited. Please try again later."
        )


# ===========================
# Multi-Pass Handler
# ===========================

def remove_thinking_tags(content: str) -> str:
    """
    Remove all <think>...</think> tags and their content from the response.
    Handles multiple occurrences and multiline content.
    """
    if not content:
        return content

    # Remove all thinking tags (not just the first one)
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)

    # Clean up extra whitespace left behind
    cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned)  # Multiple blank lines to double

    return cleaned.strip()


class MultiPassHandler:
    """Handles multi-pass request processing and response merging."""

    def __init__(self, api_client: CerebrasApiClient, api_key_manager: ApiKeyManager):
        self.api_client = api_client
        self.api_key_manager = api_key_manager

    async def handle_multi_pass_request(
        self,
        request: Request,
        incoming_data: Dict[str, Any],
        model_pool: List[str],
        pass_count: int
    ) -> JSONResponse:
        """Handle multi-pass workflow with concurrent requests and merging."""

        if incoming_data.get("stream", False):
            logger.warning(f"Streaming not supported with multi-pass mode. Disabling streaming.")
            incoming_data["stream"] = False

        if len(model_pool) < pass_count:
            raise HTTPException(
                status_code=500,
                detail=f"Multi-pass mode requires at least {pass_count} models, but pool has {len(model_pool)}"
            )

        # Select random models for this pass
        selected_models = random.sample(model_pool, pass_count)
        logger.info(f"Multi-pass: Calling models {selected_models}")

        # Execute concurrent requests
        tasks = []
        for model in selected_models:
            data = incoming_data.copy()
            data["model"] = model
            tasks.append(self.api_client.forward_request(request, data))

        # Wait for all responses
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process responses
        successful_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Multi-pass leg {i} failed: {response}")
                continue

            if isinstance(response, httpx.Response):
                successful_responses.append(response)

        if not successful_responses:
            raise HTTPException(
                status_code=502,
                detail=f"All {pass_count} multi-pass requests failed"
            )

        # Extract content from responses
        response_contents = []
        for response in successful_responses:
            try:
                response_json = response.json()
                content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    # Remove thinking tags
                    content = remove_thinking_tags(content)
                    response_contents.append(content)
            except Exception as e:
                logger.error(f"Failed to parse multi-pass response: {e}")
            finally:
                await response.aclose()

        if not response_contents:
            raise HTTPException(
                status_code=502,
                detail="Could not extract content from any multi-pass responses"
            )

        # Create merge prompt
        merge_prompt = self._create_merge_prompt(incoming_data, response_contents)

        # Use random merging model
        merge_model = random.choice(Settings.MERGING_MODELS)
        logger.info(f"Multi-pass: Merging with model '{merge_model}'")

        merge_data = {
            "model": merge_model,
            "messages": [{"role": "user", "content": merge_prompt}],
            "stream": False,
            "top_p": incoming_data.get("top_p", 0.9),
            "temperature": incoming_data.get("temperature", 0.05),
            "max_tokens": incoming_data.get("max_tokens", 8192),
        }

        # Get final merged response
        final_response = await self.api_client.forward_request(request, merge_data)
        final_response_dict = final_response.json()

        # Remove thinking tags from final response
        if "choices" in final_response_dict:
            for choice in final_response_dict["choices"]:
                if "message" in choice and "content" in choice["message"]:
                    choice["message"]["content"] = remove_thinking_tags(choice["message"]["content"])

        await final_response.aclose()
        return JSONResponse(content=final_response_dict)

    def _create_merge_prompt(self, original_request: Dict[str, Any], responses: List[str]) -> str:
        """Create the merge prompt for combining multiple responses."""
        original_messages_str = json.dumps(original_request["messages"], indent=1)
        half_of_original_messages_str = original_messages_str[len(original_messages_str) // 3:]

        draft_responses_str = ""
        for i, content in enumerate(responses):
            draft_letter = chr(ord("A") + i)
            draft_responses_str += f"Draft Response {draft_letter}:\n---\n{content}\n---\n\n"

        return f"""The original history:
```json
{half_of_original_messages_str}
```

I have received {len(responses)} different draft responses to the last message from different AI assistants.

{draft_responses_str.strip()}

Your task is to synthesize these draft responses into a single, coherent, and high-quality final response. Combine the strengths of all drafts, remove any redundancies, and present the final, polished response as if you were continuing the conversation.
So instead of this two answers the one merged answer will be used.
CAREFULLY MERGE TOOLS CALLS."""





# ===========================
# FastAPI Application
# ===========================

@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """Application lifespan manager."""
    # Initialize components
    app.state.api_key_manager = ApiKeyManager()
    app.state.api_key_manager.load_api_keys()

    app.state.cerebras_client = CerebrasApiClient(app.state.api_key_manager)
    app.state.multi_pass_handler = MultiPassHandler(app.state.cerebras_client, app.state.api_key_manager)

    # HTTP client
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(Constants.DEFAULT_TIMEOUT),
        limits=httpx.Limits(
            max_keepalive_connections=Constants.MAX_KEEPALIVE_CONNECTIONS,
            max_connections=Constants.MAX_CONNECTIONS
        )
    )

    # Log configuration
    if Settings.CLEVER_PASS_COUNT > 1:
        logger.info(f"Multi-pass mode enabled for /clever endpoint ({Settings.CLEVER_PASS_COUNT} passes)")

    logger.info("Cerebras proxy is ready!")

    yield

    # Cleanup
    await app.state.http_client.aclose()


async def verify_api_key(authorization: Optional[str] = Header(None)) -> None:
    """Verify API key from authorization header."""
    if Settings.API_KEY and (
        not authorization or
        not authorization.startswith("Bearer ") or
        authorization[7:] != Settings.API_KEY
    ):
        raise HTTPException(status_code=401, detail="Invalid API key")


app = FastAPI(
    title="Cerebras API Key Rotator",
    description="OpenAI-compatible proxy for Cerebras with key rotation and multi-pass support",
    lifespan=lifespan
)


# ===========================
# API Endpoints
# ===========================

@app.get("/cerebras/v1/models")
async def list_cerebras_models(authorization: Optional[str] = Header(None)):
    """List available Cerebras models."""
    await verify_api_key(authorization)

    return {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "owned_by": "cerebras"}
            for model in Settings.ALL_MODELS
        ]
    }


@app.post("/v1/chat/completions")
async def proxy_to_cerebras(request: Request, authorization: Optional[str] = Header(None)):
    """Standard Cerebras proxy endpoint."""
    await verify_api_key(authorization)
    return await handle_proxy_request(request, Settings.ALL_MODELS, pass_count=1)


@app.post("/cerebras/v1/chat/completions")
async def proxy_to_cerebras_explicit(request: Request, authorization: Optional[str] = Header(None)):
    """Explicit Cerebras endpoint."""
    await verify_api_key(authorization)
    return await handle_proxy_request(request, Settings.ALL_MODELS, pass_count=1)


@app.post("/clever/chat/completions")
async def proxy_to_cerebras_clever(request: Request, authorization: Optional[str] = Header(None)):
    """Multi-pass Cerebras endpoint for enhanced responses."""
    await verify_api_key(authorization)
    return await handle_proxy_request(request, Settings.CLEVER_MODELS, pass_count=Settings.CLEVER_PASS_COUNT)


async def handle_proxy_request(
    request: Request,
    model_pool: List[str],
    pass_count: int = 1
) -> Union[StreamingResponse, JSONResponse]:
    """Core request handler with multi-pass support."""

    # Handle standard Cerebras request
    try:
        incoming_data = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")

    # Process message content
    if "messages" in incoming_data and isinstance(incoming_data["messages"], list):
        for message in incoming_data["messages"]:
            if "content" in message and isinstance(message["content"], list):
                new_content_parts = []
                for part in message["content"]:
                    if isinstance(part, dict) and part.get("type") == "text":
                        new_content_parts.append(part.get("text", ""))
                message["content"] = "\n".join(new_content_parts)

    # Set defaults
    incoming_data.setdefault("top_p", 0.9)
    incoming_data.setdefault("temperature", 0.05)
    incoming_data.setdefault("max_tokens", 8192)

    # Handle multi-pass requests
    if pass_count > 1:
        return await app.state.multi_pass_handler.handle_multi_pass_request(
            request, incoming_data, model_pool, pass_count
        )

    # Handle single-pass requests
    # Use requested model, or pick random if not specified or explicitly "random"
    requested_model = incoming_data.get("model", "random")
    if requested_model == "random" or not requested_model:
        incoming_data["model"] = random.choice(model_pool)
        logger.info(f"Selected random model: {incoming_data['model']} (from pool of {len(model_pool)})")
    else:
        # Use the requested model
        if requested_model not in Settings.ALL_MODELS:
            logger.warning(f"Requested model '{requested_model}' not in available models. Using random model.")
            incoming_data["model"] = random.choice(model_pool)
        else:
            incoming_data["model"] = requested_model
            logger.info(f"Using requested model: {incoming_data['model']}")

    # Set model-specific parameters
    if incoming_data["model"] == "gpt-oss-120b":
        incoming_data.setdefault("reasoning_effort", "high")
    else:
        incoming_data.pop("reasoning_effort", None)

    try:
        response = await app.state.cerebras_client.forward_request(request, incoming_data)
        response_dict = response.json()

        # Remove thinking tags from all response content
        if "choices" in response_dict:
            for choice in response_dict["choices"]:
                if "message" in choice and "content" in choice["message"]:
                    choice["message"]["content"] = remove_thinking_tags(choice["message"]["content"])

        return JSONResponse(content=response_dict)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ===========================
# Health Check
# ===========================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "cerebras_keys": len(Settings.CEREBRAS_KEYS),
        "models_available": len(Settings.ALL_MODELS),
        "multi_pass_enabled": Settings.CLEVER_PASS_COUNT > 1,
        "timestamp": int(time.time())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)
