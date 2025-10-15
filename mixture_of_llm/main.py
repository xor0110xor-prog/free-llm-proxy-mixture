# aggregator_proxy_final_unified.py

# A definitive, unified FastAPI proxy server that aggregates responses from multiple LLMs
# and uses a SINGLE-STEP Mixture-of-Agents (MOA) to synthesize a final answer.
# NOW WITH EXTERNAL CONFIGURATION VIA config.yaml
# ENHANCED WITH MULTI-MASTER AGENT RACING
# ENHANCED WITH FULL END-TO-END TOOL_CALLS SUPPORT

import asyncio
import re
import time
import uuid
import random
import json
from contextlib import asynccontextmanager
from typing import Dict, Any, AsyncGenerator, List, Set, Optional
from pathlib import Path
import httpx
import yaml
from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
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


# Settings class to load API key from config
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

    # API Key for authentication
    API_KEY: Optional[str] = config.get("api_key")


# Load global configuration
CONFIG = load_config()

# Extract configuration sections
TARGET_APIS = CONFIG['downstream_apis']
MOA_CONFIG = CONFIG['moa']
TIMEOUT_CONFIG = CONFIG['timeouts']
RETRY_CONFIG = CONFIG['retry']
SERVER_CONFIG = CONFIG['server']
PROMPTS = CONFIG['prompts']

# --- Extracted Configuration Variables ---
MOA_MASTER_AGENT_KEY = MOA_CONFIG['master_agent_key']
MOA_MASTER_AGENT_KEYS = MOA_CONFIG.get('master_agent_keys', [MOA_MASTER_AGENT_KEY])
MOA_RACING_ENABLED = MOA_CONFIG.get('racing_enabled', False)
MOA_RACING_RUNS = MOA_CONFIG.get('racing_runs', 1)

NUM_CANDIDATES = MOA_CONFIG['num_candidates']
FASTER_PLUS_ONE = MOA_CONFIG['faster_plus_one']
FASTER_PLUS_NUM = MOA_CONFIG['faster_plus_num']
MINIMUM_CANDIDATES = MOA_CONFIG['minimum_candidates']
MOA_ENABLE_DETAILED_LOGGING = MOA_CONFIG['enable_detailed_logging']
MOA_MAX_CANDIDATE_LENGTH = MOA_CONFIG['max_candidate_length']

AGGREGATION_SOFT_TIMEOUT_SECONDS = TIMEOUT_CONFIG['soft_timeout_seconds']
AGGREGATION_HARD_TIMEOUT_SECONDS = TIMEOUT_CONFIG['hard_timeout_seconds']
HTTP_CLIENT_TIMEOUT = TIMEOUT_CONFIG['http_client_timeout']

MAX_RETRIES = RETRY_CONFIG['max_retries']
BASE_DELAY = RETRY_CONFIG['base_delay']
MAX_DELAY = RETRY_CONFIG['max_delay']
JITTER_RANGE = RETRY_CONFIG['jitter_range']

MOA_SYSTEM_PROMPT = PROMPTS['moa_system_prompt']
MOA_UNIFIED_SYNTHESIS_PROMPT = PROMPTS['moa_unified_synthesis_prompt']
CANDIDATE_FORMAT_TEMPLATE = PROMPTS['candidate_format_template']


# --- FastAPI Application Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages the lifecycle of the shared HTTP client."""
    app.state.http_client = httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT)
    yield
    await app.state.http_client.aclose()


app = FastAPI(
    title="Definitive MOA Aggregator Proxy with Multi-Master Racing & Tool Calls",
    description="Aggregates LLM responses with tool_calls support, multi-master agent racing and uses a single-step MOA for synthesis.",
    lifespan=lifespan
)


# --- Authentication Functions ---
async def verify_api_key(authorization: Optional[str] = Header(None)) -> None:
    """Verify API key from authorization header."""
    if Settings.API_KEY and (
        not authorization or
        not authorization.startswith("Bearer ") or
        authorization[7:] != Settings.API_KEY
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
    if isinstance(error, (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in [429, 502, 503, 504] or 500 <= error.response.status_code < 600
    return False


async def call_downstream_api_with_retry(
        client: httpx.AsyncClient,
        target_config: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
        max_retries: int = MAX_RETRIES
) -> Dict[str, Any]:
    """
    Asynchronously calls a single downstream API with robust retry logic and tool support.
    """
    url = target_config["base_url"].rstrip('/') + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {target_config['api_key']}"}

    payload = {
        "model": target_config["model"],
        "messages": messages,
        "stream": target_config.get("stream", False),
    }

    # Add tool-related parameters if provided
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = parallel_tool_calls

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                delay = await exponential_backoff_delay(attempt - 1)
                await asyncio.sleep(delay)

            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()

        except Exception as error:
            last_error = error
            if attempt < max_retries and is_retryable_error(error):
                logger.warning(f"Retryable error calling {url} (attempt {attempt + 1}/{max_retries + 1}): {error}")
            else:
                break

    error_details = str(last_error.response.text) if isinstance(last_error, httpx.HTTPStatusError) else str(last_error)
    return {"error": f"API call failed after {attempt + 1} attempts", "details": error_details,
            "retry_attempts": attempt}


# --- Core Logic: Aggregation with Two-Tier Timeout ---
async def process_aggregate_request(num_targets: int, request: Request, faster_plus_one: bool = False) -> Dict[
    str, Any]:
    """
    Gathers responses from multiple LLMs, preserving full message objects including tool_calls.
    """
    request_id = f"aggregator-{uuid.uuid4()}"
    start_time = time.time()

    try:
        data = await request.json()
        messages = data.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="Parameter 'messages' is required.")

        # Extract tool-calling parameters to pass to downstream APIs
        tools = data.get("tools")
        tool_choice = data.get("tool_choice")
        parallel_tool_calls = data.get("parallel_tool_calls")

        client: httpx.AsyncClient = request.app.state.http_client
        num_to_request = num_targets + FASTER_PLUS_NUM if faster_plus_one else num_targets

        logger.info(
            f"[{request_id}] Mode: {'FASTER_PLUS_ONE' if faster_plus_one else 'Standard'}. "
            f"Requesting {num_to_request}, using first {num_targets}. "
            f"Min Candidates: {MINIMUM_CANDIDATES}, Soft Timeout: {AGGREGATION_SOFT_TIMEOUT_SECONDS}s, Hard Timeout: {AGGREGATION_HARD_TIMEOUT_SECONDS}s. "
            f"Tools in request: {tools is not None}"
        )

        targets_to_call = list(TARGET_APIS.items())[:num_to_request]
        tasks = [
            asyncio.create_task(
                call_downstream_api_with_retry(
                    client, config, messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    parallel_tool_calls=parallel_tool_calls
                )
            )
            for _, config in targets_to_call
        ]
        task_to_api_id = {task: api_id for task, (api_id, _) in zip(tasks, targets_to_call)}

        results = []
        pending_tasks: Set[asyncio.Task] = set(tasks)
        in_overtime = False
        timeout_reason = "none"

        while pending_tasks:
            elapsed_time = time.time() - start_time

            is_target_met = len(results) >= num_targets
            is_minimum_met = len(results) >= MINIMUM_CANDIDATES
            is_soft_timeout_reached = elapsed_time >= AGGREGATION_SOFT_TIMEOUT_SECONDS
            is_hard_timeout_reached = elapsed_time >= AGGREGATION_HARD_TIMEOUT_SECONDS

            if is_target_met:
                logger.info(
                    f"[{request_id}] Target of {num_targets} candidates reached in {elapsed_time:.2f}s. Finishing early.")
                timeout_reason = "target_met"
                break
            if is_hard_timeout_reached:
                logger.warning(
                    f"[{request_id}] Hard timeout of {AGGREGATION_HARD_TIMEOUT_SECONDS}s reached. Stopping with {len(results)} candidates.")
                timeout_reason = "hard_timeout"
                break
            if is_soft_timeout_reached and is_minimum_met:
                logger.info(
                    f"[{request_id}] Soft timeout of {AGGREGATION_SOFT_TIMEOUT_SECONDS}s reached with {len(results)} (>= minimum {MINIMUM_CANDIDATES}) candidates. Proceeding.")
                timeout_reason = "soft_timeout"
                break
            if is_soft_timeout_reached and not in_overtime:
                logger.warning(
                    f"[{request_id}] Soft timeout reached, but only have {len(results)}/{MINIMUM_CANDIDATES} candidates. Extending wait until hard timeout.")
                in_overtime = True

            remaining_time = AGGREGATION_HARD_TIMEOUT_SECONDS - elapsed_time
            done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED,
                                                     timeout=max(0.1, remaining_time))
            if not done:
                continue

            for completed_task in done:
                api_id = task_to_api_id.get(completed_task, "unknown")
                result = await completed_task
                results.append({"api_id": api_id, "response": result})
                elapsed = time.time() - start_time
                logger.info(f"[{request_id}] Response #{len(results)} received from {api_id} in {elapsed:.2f}s.")

        if pending_tasks:
            cancelled_apis = [task_to_api_id.get(task, "unknown") for task in pending_tasks]
            logger.warning(
                f"[{request_id}] Cancelling {len(pending_tasks)} remaining tasks due to '{timeout_reason}': {', '.join(cancelled_apis)}")
            for task in pending_tasks:
                task.cancel()
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        # Process results, preserving the full message object
        choices = []
        valid_responses_count = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        all_tool_calls = []

        for i, res in enumerate(results[:num_targets]):
            api_id = res["api_id"]
            result = res["response"]

            if "error" in result:
                message = {"role": "assistant",
                           "content": f"ERROR from {api_id}: {result.get('details', result['error'])}"}
                finish_reason = "error"
            else:
                valid_responses_count += 1
                choice = result.get("choices", [{}])[0]
                message = choice.get("message", {"role": "assistant", "content": ""})
                finish_reason = choice.get("finish_reason", "stop")
                if message.get("tool_calls"):
                    all_tool_calls.extend(message["tool_calls"])

                usage = result.get("usage", {})
                total_prompt_tokens += usage.get("prompt_tokens", 0)
                total_completion_tokens += usage.get("completion_tokens", 0)

            choices.append({"index": i, "message": message, "finish_reason": finish_reason})

        total_elapsed = time.time() - start_time
        logger.info(f"[{request_id}] Aggregation complete. Collected {len(all_tool_calls)} tool calls from candidates.")

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": f"aggregator-v4-n{num_targets}-min{MINIMUM_CANDIDATES}",
            "choices": choices,
            "usage": {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens,
                      "total_tokens": total_prompt_tokens + total_completion_tokens},
            "metadata": {
                "total_elapsed_seconds": round(total_elapsed, 2),
                "requested_targets": num_targets,
                "completed_targets": len(results),
                "valid_targets": valid_responses_count,
                "minimum_candidates": MINIMUM_CANDIDATES,
                "soft_timeout_seconds": AGGREGATION_SOFT_TIMEOUT_SECONDS,
                "hard_timeout_seconds": AGGREGATION_HARD_TIMEOUT_SECONDS,
                "termination_reason": timeout_reason,
                "extended_for_minimum_candidates": in_overtime,
                "tool_calls_from_candidates": len(all_tool_calls)
            }
        }

    except Exception as e:
        logger.error(f"[{request_id}] Critical error in aggregator: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal aggregator error: {e}")



def format_candidate_for_synthesis(candidate_message: Dict[str, Any]) -> str:
    """Formats a candidate message into a string for the master agent's prompt."""
    content = candidate_message.get("content", "")
    tool_calls = candidate_message.get("tool_calls")
    parts = []

    if content:
        parts.append(f"Content:\n{content}")

    if tool_calls:
        # Full serialization instead of compact summary
        try:
            tool_calls_str = json.dumps(tool_calls, indent=2, ensure_ascii=False)
        except Exception:
            tool_calls_str = str(tool_calls)
        parts.append(f"Tool Calls:\n{tool_calls_str}")

    formatted_str = "\n\n".join(parts)
    return formatted_str


# --- Core Logic: Mixture-of-Agents Synthesis ---
async def async_unified_moa(
        initial_query: str,
        candidates: List[Dict[str, Any]],
        client: httpx.AsyncClient,
        master_api_config: Dict[str, Any],
        original_tools: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Performs MOA synthesis, handling candidates with tool_calls and allowing the master agent to generate its own.
    """
    logger.info(f"Starting unified MOA synthesis with master agent: {master_api_config['model']}")

    candidates_section = "\n".join(
        [CANDIDATE_FORMAT_TEMPLATE.format(index=i + 1, content=format_candidate_for_synthesis(c)) for i, c in
         enumerate(candidates)]
    )


    synthesis_prompt = MOA_UNIFIED_SYNTHESIS_PROMPT.format(
        initial_query=initial_query,
        candidates_section=candidates_section
    )
    synthesis_messages = [
        {"role": "system", "content": MOA_SYSTEM_PROMPT},
        {"role": "user", "content": synthesis_prompt}
    ]

    llm_response_json = await call_downstream_api_with_retry(
        client,
        master_api_config,
        synthesis_messages,
        tools=original_tools
    )

    if "error" in llm_response_json:
        raise HTTPException(status_code=502, detail=f"MOA synthesis step failed: {llm_response_json['error']}")

    choice = llm_response_json.get("choices", [{}])[0]
    message = choice.get("message", {})

    # If the master agent returns tool_calls, we prioritize that and return.
    if message.get("tool_calls"):
        logger.info(f"Master agent returned {len(message['tool_calls'])} tool_calls. Returning directly.")
        return llm_response_json

    # Otherwise, process for a final text answer.
    full_llm_output = message.get("content", "")
    if not full_llm_output:
        logger.error(f"Master agent returned empty content: {llm_response_json}")
        # await asyncio.sleep(30)  # let others continue first
        # return 'error'
        raise HTTPException(status_code=502, detail="MOA master agent returned empty content and no tool_calls.")

    match = re.search(r"<final_answer>(.*?)</final_answer>", full_llm_output, re.DOTALL)
    if match:
        final_content = match.group(1).strip()
    else:
        logger.warning("Could not find <final_answer> tag. Using full LLM output as fallback.")
        final_content = full_llm_output


    llm_response_json["choices"][0]["message"]["content"] = final_content
    return llm_response_json


# --- Multi-Master Agent Racing Logic ---
async def race_master_agents(
        initial_query: str,
        candidates: List[Dict[str, Any]],
        client: httpx.AsyncClient,
        original_tools: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Races multiple master agents and returns the fastest valid response."""
    if not MOA_RACING_ENABLED or len(MOA_MASTER_AGENT_KEYS) <= 1:
        master_api_config = TARGET_APIS[MOA_MASTER_AGENT_KEY]
        return await async_unified_moa(initial_query, candidates, client, master_api_config, original_tools)

    race_id = f"race-{uuid.uuid4()}"
    logger.info(
        f"[{race_id}] Starting master agent racing with {len(MOA_MASTER_AGENT_KEYS)} agents, {MOA_RACING_RUNS} runs each"
    )

    racing_tasks = []
    task_metadata = []
    for master_key in MOA_MASTER_AGENT_KEYS:
        if master_key not in TARGET_APIS:
            logger.warning(f"[{race_id}] Master agent key '{master_key}' not found in TARGET_APIS, skipping")
            continue
        master_config = TARGET_APIS[master_key]
        for run_idx in range(MOA_RACING_RUNS):
            task = asyncio.create_task(
                async_unified_moa(initial_query, candidates, client, master_config, original_tools)
            )
            racing_tasks.append(task)
            task_metadata.append({
                "master_key": master_key,
                "master_model": master_config.get("model", "unknown"),
                "run_index": run_idx,
                "task": task
            })

    if not racing_tasks:
        raise HTTPException(status_code=500, detail="No valid master agents found for racing")

    logger.info(f"[{race_id}] Racing {len(racing_tasks)} total tasks")
    start_time = time.time()

    winner_result = None
    winner_metadata = None
    pending_tasks = set(racing_tasks)

    try:
        # Keep waiting until we find a valid result or all tasks fail
        while pending_tasks and winner_result is None:
            done, pending_tasks = await asyncio.wait(
                pending_tasks,
                return_when=asyncio.FIRST_COMPLETED
            )

            for completed_task in done:
                try:
                    result = await completed_task

                    # Skip if error
                    if "error" in result:
                        for meta in task_metadata:
                            if meta["task"] == completed_task:
                                logger.warning(
                                    f"[{race_id}] Master agent {meta['master_key']} (run {meta['run_index']}) failed: {result.get('details', result['error'])}"
                                )
                        continue  # Try next completed task in done set

                    # Found winner!
                    for meta in task_metadata:
                        if meta["task"] == completed_task:
                            winner_metadata = meta
                            break
                    winner_result = result
                    break  # Exit for loop

                except Exception as e:
                    logger.warning(f"[{race_id}] Racing task failed with exception: {e}")
                    continue

        # Cancel remaining tasks
        if pending_tasks:
            logger.info(f"[{race_id}] Cancelling {len(pending_tasks)} remaining racing tasks")
            for task in pending_tasks:
                task.cancel()
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        if winner_result is None:
            raise HTTPException(status_code=502, detail="All master agents failed in racing mode")

        elapsed_time = time.time() - start_time
        logger.info(
            f"[{race_id}] Winner: {winner_metadata['master_key']} ({winner_metadata['master_model']}) run {winner_metadata['run_index']} in {elapsed_time:.2f}s"
        )

        if "metadata" not in winner_result:
            winner_result["metadata"] = {}

        winner_result["metadata"]["racing_winner"] = winner_metadata["master_key"]
        winner_result["metadata"]["racing_winner_model"] = winner_metadata["master_model"]
        winner_result["metadata"]["racing_run_index"] = winner_metadata["run_index"]
        winner_result["metadata"]["racing_elapsed_seconds"] = round(elapsed_time, 2)
        winner_result["metadata"]["racing_total_tasks"] = len(racing_tasks)

        return winner_result

    except Exception as e:
        # Emergency cleanup: cancel all tasks if something goes wrong
        for task in racing_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*racing_tasks, return_exceptions=True)
        logger.error(f"[{race_id}] Racing failed with error: {e}")
        raise HTTPException(status_code=502, detail=f"Master agent racing failed: {e}")


# --- API Endpoints ---
async def process_moa_request(request: Request, num_candidates: int) -> JSONResponse:
    """General handler for all MOA requests with full tool_calls support."""
    request_id = f"moa-{uuid.uuid4()}"
    logger.info(f"[{request_id}] Received MOA request for N={num_candidates} candidates.")
    try:
        aggregated_data = await process_aggregate_request(num_targets=num_candidates, request=request,
                                                          faster_plus_one=FASTER_PLUS_ONE)

        # Filter for valid candidates (not errors)
        candidates = [choice['message'] for choice in aggregated_data.get('choices', []) if
                      not choice['message'].get('content', '').startswith("ERROR from")]

        if len(candidates) < MINIMUM_CANDIDATES:
            raise HTTPException(status_code=502,
                                detail=f"Failed to gather minimum required candidates. Got {len(candidates)}, required {MINIMUM_CANDIDATES}.")

        logger.warning(f"[{request_id}] Gathered {len(candidates)} valid candidates for MOA synthesis. {candidates=}")

        data = await request.json()
        initial_query = json.dumps(data.get("messages", []))
        original_tools = data.get("tools")
        client: httpx.AsyncClient = request.app.state.http_client

        if MOA_RACING_ENABLED and len(MOA_MASTER_AGENT_KEYS) > 1:
            final_moa_result = await race_master_agents(initial_query, candidates, client, original_tools)
        else:
            master_api_config = TARGET_APIS[MOA_MASTER_AGENT_KEY]
            final_moa_result = await async_unified_moa(initial_query, candidates, client, master_api_config,
                                                       original_tools)

        master_usage = final_moa_result.get("usage", {})
        logger.info(f"[{request_id}] MOA COMPLETE. Master agent tokens: total={master_usage.get('total_tokens', 0)}")

        final_moa_result["id"] = request_id
        final_moa_result["model"] = data['model']

        logger.debug(f"[{request_id}] Final MOA result: {json.dumps(final_moa_result, indent=2)}")
        logger.warning(f"[{request_id}] Returning final MOA response: {final_moa_result=}")
        return JSONResponse(content=final_moa_result)
    except Exception as e:
        if isinstance(e, HTTPException): raise
        logger.error(f"[{request_id}] Error in unified MOA process: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unified MOA processing error: {e}")


# --- Random API Selection Logic ---
async def call_random_api(
        request: Request,
        num_apis: int = 1,
        race_mode: bool = False
) -> JSONResponse:
    """
    Calls random downstream API(s) and optionally races them for fastest response.

    Args:
        request: FastAPI request object
        num_apis: Number of random APIs to call (default 1)
        race_mode: If True, races multiple APIs and returns fastest result
    """
    request_id = f"random-{uuid.uuid4()}"
    start_time = time.time()

    try:
        data = await request.json()
        messages = data.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="Parameter 'messages' is required.")

        # Extract tool-calling parameters
        tools = data.get("tools")
        tool_choice = data.get("tool_choice")
        parallel_tool_calls = data.get("parallel_tool_calls")

        client: httpx.AsyncClient = request.app.state.http_client

        # Select random APIs
        available_apis = list(TARGET_APIS.items())
        if num_apis > len(available_apis):
            num_apis = len(available_apis)
            logger.warning(f"[{request_id}] Requested {num_apis} APIs, but only {len(available_apis)} available")

        selected_apis = random.sample(available_apis, num_apis)
        logger.info(f"[{request_id}] Selected random APIs: {[api_id for api_id, _ in selected_apis]}")

        if not race_mode or num_apis == 1:
            # Single API call mode
            api_id, api_config = selected_apis[0]
            logger.info(f"[{request_id}] Calling single random API: {api_id}")

            result = await call_downstream_api_with_retry(
                client, api_config, messages,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls
            )

            if "error" in result:
                raise HTTPException(
                    status_code=502,
                    detail=f"API {api_id} failed: {result.get('details', result['error'])}"
                )

            elapsed_time = time.time() - start_time
            result["id"] = request_id
            result["model"] = f"random-{api_id}"
            if "metadata" not in result:
                result["metadata"] = {}
            result["metadata"]["selected_api"] = api_id
            result["metadata"]["elapsed_seconds"] = round(elapsed_time, 2)
            result["metadata"]["mode"] = "single"

            logger.info(f"[{request_id}] Single API call completed in {elapsed_time:.2f}s")
            return JSONResponse(content=result)

        else:
            # Racing mode - call all selected APIs and return fastest
            logger.info(f"[{request_id}] Racing {num_apis} random APIs")

            tasks = []
            task_metadata = []

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

            winner_result = None
            winner_metadata = None

            # Wait for first successful completion
            pending_tasks = set(tasks)
            while pending_tasks and winner_result is None:
                done, pending_tasks = await asyncio.wait(
                    pending_tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )

                for completed_task in done:
                    try:
                        result = await completed_task

                        # Skip if error
                        if "error" in result:
                            for meta in task_metadata:
                                if meta["task"] == completed_task:
                                    logger.warning(
                                        f"[{request_id}] API {meta['api_id']} failed: {result.get('details', result['error'])}"
                                    )
                            continue

                        # Found winner!
                        for meta in task_metadata:
                            if meta["task"] == completed_task:
                                winner_metadata = meta
                                break
                        winner_result = result
                        break

                    except Exception as e:
                        logger.warning(f"[{request_id}] Racing task failed with exception: {e}")
                        continue

            # Cancel remaining tasks
            if pending_tasks:
                logger.info(f"[{request_id}] Cancelling {len(pending_tasks)} remaining racing tasks")
                for task in pending_tasks:
                    task.cancel()
                await asyncio.gather(*pending_tasks, return_exceptions=True)

            if winner_result is None:
                raise HTTPException(
                    status_code=502,
                    detail="All random APIs failed in racing mode"
                )

            elapsed_time = time.time() - start_time
            winner_result["id"] = request_id
            winner_result["model"] = f"fastest-{winner_metadata['api_id']}"

            if "metadata" not in winner_result:
                winner_result["metadata"] = {}

            winner_result["metadata"]["winner_api"] = winner_metadata["api_id"]
            winner_result["metadata"]["winner_model"] = winner_metadata["model"]
            winner_result["metadata"]["elapsed_seconds"] = round(elapsed_time, 2)
            winner_result["metadata"]["mode"] = "racing"
            winner_result["metadata"]["num_racers"] = num_apis
            winner_result["metadata"]["tested_apis"] = [meta["api_id"] for meta in task_metadata]

            logger.info(
                f"[{request_id}] Racing winner: {winner_metadata['api_id']} "
                f"({winner_metadata['model']}) in {elapsed_time:.2f}s"
            )

            return JSONResponse(content=winner_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error in random API call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Random API call error: {e}")


# --- Random API Endpoints ---
@app.post("/random_v1/chat/completions")
async def random_single_api(request: Request) -> JSONResponse:
    """
    Selects and calls a single random downstream API.
    Useful for load distribution and testing different models.
    """
    return await call_random_api(request, num_apis=1, race_mode=False)


@app.post("/v1/chat/completions")
async def moa_endpoint(request: Request, authorization: Optional[str] = Header(None)) -> JSONResponse:
    await verify_api_key(authorization)
    # body = await request.json()
    # logger.debug(f"Received /v1/chat/completions request body: {json.dumps(body, indent=2)}")
    return await process_moa_request(request, num_candidates=NUM_CANDIDATES)


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None)) -> JSONResponse:
    """Lists available models (downstream APIs)."""
    await verify_api_key(authorization)
    models = [{"id": api_id, "object": "model", "owned_by": "aggregator", "model_info": config} for api_id, config in
              TARGET_APIS.items()]
    return JSONResponse(content={"object": "list", "data": models})


@app.get("/health")
async def health_check(authorization: Optional[str] = Header(None)):
    """Provides a health check and system configuration overview."""
    await verify_api_key(authorization)
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "system_config": {
            "synthesis_agent": TARGET_APIS.get(MOA_MASTER_AGENT_KEY, {}).get("model", "unknown"),
            "racing_enabled": MOA_RACING_ENABLED,
            "racing_masters": MOA_MASTER_AGENT_KEYS if MOA_RACING_ENABLED else [MOA_MASTER_AGENT_KEY],
            "racing_runs_per_master": MOA_RACING_RUNS,
            "faster_plus_one_enabled": FASTER_PLUS_ONE,
            "max_retries": MAX_RETRIES,
            "tool_calls_supported": True
        },
        "aggregation_config": {
            "minimum_candidates": MINIMUM_CANDIDATES,
            "soft_timeout_seconds": AGGREGATION_SOFT_TIMEOUT_SECONDS,
            "hard_timeout_seconds": AGGREGATION_HARD_TIMEOUT_SECONDS,
        },
        "available_apis": list(TARGET_APIS.keys())
    }


# --- Main Execution Block ---
if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting Definitive Unified FastAPI Aggregator Proxy Server on port {SERVER_CONFIG['port']}...")
    if MOA_RACING_ENABLED and len(MOA_MASTER_AGENT_KEYS) > 1:
        logger.info(f"MOA Racing Mode ENABLED: {len(MOA_MASTER_AGENT_KEYS)} master agents, {MOA_RACING_RUNS} runs each")
        logger.info(f"Racing Masters: {[TARGET_APIS.get(k, {}).get('model', k) for k in MOA_MASTER_AGENT_KEYS]}")
    else:
        logger.info(f"MOA Single Master Mode: {TARGET_APIS.get(MOA_MASTER_AGENT_KEY, {}).get('model', 'unknown')}")
    logger.info(f"Performance Config: FASTER_PLUS_ONE: {FASTER_PLUS_ONE}")
    logger.info(
        f"Aggregation Policy: Min {MINIMUM_CANDIDATES} candidates, Soft Timeout {AGGREGATION_SOFT_TIMEOUT_SECONDS}s, Hard Timeout {AGGREGATION_HARD_TIMEOUT_SECONDS}s")
    logger.info("Tool Calls Support: ENABLED")
    uvicorn.run(app, host=SERVER_CONFIG['host'], port=SERVER_CONFIG['port'])
