import asyncio
import json
import os
import random
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
    """Centralized constants for Groq configuration."""

    # API Configuration
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    RATE_LIMIT_COOLDOWN = 120  # Base cooldown in seconds
    STATE_FILE = "groq_api_key_state.json"

    # Request configuration
    REQUEST_TIMEOUT = 300.0
    DEFAULT_TIMEOUT = 30.0
    MAX_KEEPALIVE_CONNECTIONS = 20
    MAX_CONNECTIONS = 100

    # Rate limit status codes
    RATE_LIMIT_STATUS_CODES = {429, 413, 500}


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

    # Groq Configuration
    GROQ_KEYS: List[str] = config.get("groq", {}).get("keys", [])
    GROQ_RATE_LIMIT_COOLDOWN: int = config.get("groq", {}).get("rate_limit_cooldown", Constants.RATE_LIMIT_COOLDOWN)
    GROQ_STATE_FILE: str = config.get("groq", {}).get("state_file", Constants.STATE_FILE)
    GROQ_API_URL: str = config.get("groq", {}).get("api_url", Constants.GROQ_API_URL)

    # Model Configuration
    AVAILABLE_MODELS: List[str] = config.get("groq", {}).get("available_models", ["compound-beta"])
    DEFAULT_MODEL: str = config.get("groq", {}).get("default_model", "compound-beta")


# ===========================
# API Key Management
# ===========================

class ApiKeyInfo:
    """Represents an API key with its state and rate limiting information."""

    def __init__(self, key: str):
        self.key = key
        self.rate_limited_until: float = 0.0
        self.last_used: float = 0.0
        self.rate_limit_history: List[float] = []

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
        key_info.rate_limited_until = data.get("rate_limited_until", 0.0)
        key_info.last_used = data.get("last_used", 0.0)
        key_info.rate_limit_history = data.get("rate_limit_history", [])
        return key_info


class ApiKeyManager:
    """Manages API keys with rotation and rate limiting."""

    def __init__(self):
        self.api_keys: List[ApiKeyInfo] = []
        self._lock = threading.Lock()

    def load_api_keys(self):
        """Load API keys from configuration and restore state from file."""
        logger.info("Loading Groq API keys from configuration...")

        for key in Settings.GROQ_KEYS:
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
            if os.path.exists(Settings.GROQ_STATE_FILE):
                with open(Settings.GROQ_STATE_FILE, "r") as f:
                    saved_state = json.load(f)

                logger.info(f"Found state file '{Settings.GROQ_STATE_FILE}'. Restoring rate limit history...")

                for key_info in self.api_keys:
                    key_value = key_info.key
                    if key_value in saved_state:
                        state = saved_state[key_value]
                        key_info.rate_limited_until = state.get("rate_limited_until", 0.0)
                        key_info.rate_limit_history = state.get("rate_limit_history", [])
                        logger.info(f"  - Restored state for key ending in ...{key_value[-4:]}")

        except FileNotFoundError:
            logger.info(f"State file '{Settings.GROQ_STATE_FILE}' not found. Starting with fresh state.")
        except json.JSONDecodeError:
            logger.warning(f"Could not parse state file '{Settings.GROQ_STATE_FILE}'. Starting with fresh state.")
        except Exception as e:
            logger.warning(f"Unexpected error loading state file: {e}. Starting with fresh state.")

    def _save_state(self):
        """Save current state to file."""
        try:
            state_to_save = {key_info.key: key_info.to_dict() for key_info in self.api_keys}
            with open(Settings.GROQ_STATE_FILE, "w") as f:
                json.dump(state_to_save, f, indent=2)
        except IOError as e:
            logger.error(f"Could not write to state file {Settings.GROQ_STATE_FILE}: {e}")

    def get_next_api_key(self) -> Optional[ApiKeyInfo]:
        """Get the next available API key."""
        with self._lock:
            now = time.time()
            available_keys = [
                key_info for key_info in self.api_keys
                if now >= key_info.rate_limited_until
            ]

            if not available_keys:
                return None

            # Return least recently used key
            best_key = min(available_keys, key=lambda k: k.last_used)
            best_key.last_used = now

            logger.info(f"Selected API key ending in ...{best_key.key[-4:]}")
            return best_key

    def mark_key_rate_limited(self, key_info: ApiKeyInfo):
        """Mark a key as rate limited with progressive timeout."""
        with self._lock:
            now = time.time()
            ten_hours_ago = now - 36000

            # Update rate limit history - tracks hits per 10 hours
            recent_history = [
                t for t in key_info.rate_limit_history if t > ten_hours_ago
            ]
            recent_history.append(now)
            key_info.rate_limit_history = recent_history

            # Calculate cooldown with exponential backoff
            cooldown_multiplier = len(recent_history)
            current_cooldown = Settings.GROQ_RATE_LIMIT_COOLDOWN * cooldown_multiplier
            message_suffix = ""
            if cooldown_multiplier > 1:
                message_suffix = f" (cooldown increased due to {cooldown_multiplier} hits in the last 10 hours)"

            cooldown_until = now + current_cooldown
            key_info.rate_limited_until = cooldown_until
            self._save_state()

        logger.warning(
            f"Rate limit hit for key ...{key_info.key[-4:]}. "
            f"Cooling down for {current_cooldown}s.{message_suffix}"
        )


# ===========================
# Groq API Client
# ===========================

class GroqApiClient:
    """Client for Groq API operations."""

    def __init__(self, api_key_manager: ApiKeyManager):
        self.api_key_manager = api_key_manager

    async def forward_request(
        self,
        request: Request,
        request_data: Dict[str, Any]
    ) -> httpx.Response:
        """Forward a request to Groq with key rotation and retry logic."""
        http_client: httpx.AsyncClient = request.app.state.http_client

        # Try each available key
        for attempt in range(len(self.api_key_manager.api_keys)):
            api_key_info = self.api_key_manager.get_next_api_key()

            if api_key_info is None:
                logger.error(f"All {len(self.api_key_manager.api_keys)} keys are rate-limited")
                raise HTTPException(
                    status_code=429,
                    detail="All available API keys are currently rate-limited. Please try again shortly."
                )

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key_info.key}",
            }

            try:
                response = await http_client.post(
                    Settings.GROQ_API_URL,
                    headers=headers,
                    json=request_data,
                    timeout=Constants.REQUEST_TIMEOUT
                )

                # Handle rate limiting
                if response.status_code in Constants.RATE_LIMIT_STATUS_CODES:
                    logger.warning(f"Rate limit from Groq: {response.status_code}")
                    self.api_key_manager.mark_key_rate_limited(api_key_info)
                    continue

                response.raise_for_status()
                return response

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error from Groq: {e.response.status_code} - {e.response.text}")
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=f"Upstream API error: {e.response.text}"
                )
            except httpx.RequestError as e:
                logger.error(f"Network error: {e}")
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to connect to Groq API: {e}"
                )

        raise HTTPException(
            status_code=429,
            detail="All available API keys were rate-limited by the upstream service. Please try again shortly."
        )


# ===========================
# FastAPI Application
# ===========================

@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """Application lifespan manager."""
    # Initialize components
    app.state.api_key_manager = ApiKeyManager()
    app.state.api_key_manager.load_api_keys()

    app.state.groq_client = GroqApiClient(app.state.api_key_manager)

    # HTTP client
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(Constants.DEFAULT_TIMEOUT),
        limits=httpx.Limits(
            max_keepalive_connections=Constants.MAX_KEEPALIVE_CONNECTIONS,
            max_connections=Constants.MAX_CONNECTIONS
        )
    )

    logger.info("Groq proxy is ready!")

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
    title="Groq API Key Rotator",
    description="OpenAI-compatible proxy for Groq with key rotation and rate limit handling",
    lifespan=lifespan
)


# ===========================
# API Endpoints
# ===========================

@app.get("/groq/v1/models")
async def list_groq_models(authorization: Optional[str] = Header(None)):
    """List available Groq models."""
    await verify_api_key(authorization)

    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "meta-llama",
                "context_window": 131072,
            }
            for model_id in Settings.AVAILABLE_MODELS
        ],
    }


@app.post("/v1/chat/completions")
async def proxy_to_groq(request: Request, authorization: Optional[str] = Header(None)):
    """Standard Groq proxy endpoint."""
    await verify_api_key(authorization)
    return await handle_proxy_request(request)


@app.post("/groq/v1/chat/completions")
async def proxy_to_groq_explicit(request: Request, authorization: Optional[str] = Header(None)):
    """Explicit Groq endpoint."""
    await verify_api_key(authorization)
    return await handle_proxy_request(request)


async def handle_proxy_request(request: Request) -> Union[StreamingResponse, JSONResponse]:
    """Core request handler for Groq API."""

    try:
        incoming_data = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")

    # Set model
    incoming_data["model"] = incoming_data.get("model", Settings.DEFAULT_MODEL)
    if incoming_data["model"] not in Settings.AVAILABLE_MODELS:
        incoming_data["model"] = random.choice(Settings.AVAILABLE_MODELS)

    logger.info(f"Request for model: {incoming_data['model']}")

    # Set defaults
    incoming_data.setdefault("temperature", 0.7)
    incoming_data.setdefault("top_p", 0.9)
    incoming_data.setdefault("max_tokens", 8192)

    is_streaming = incoming_data.get("stream", False)

    try:
        response = await app.state.groq_client.forward_request(request, incoming_data)

        if is_streaming:
            async def stream_generator():
                async for chunk in response.aiter_bytes():
                    yield chunk

            return StreamingResponse(
                stream_generator(),
                media_type=response.headers.get("Content-Type")
            )
        else:
            return JSONResponse(content=response.json())

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
        "groq_keys": len(Settings.GROQ_KEYS),
        "models_available": len(Settings.AVAILABLE_MODELS),
        "timestamp": int(time.time())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
