"""Provider-agnostic VLM client supporting OpenAI and Anthropic.

Auto-detects the provider from the available API key
(``OPENAI_API_KEY`` or ``ANTHROPIC_API_KEY``).  The model name can also
hint at the provider (``gpt-*`` → OpenAI, ``claude-*`` → Anthropic).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_IMAGES_PER_CALL = 8
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds — exponential backoff base

_MEDIA_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}

_PRIORITY_VIEWS = [
    "scene_default",
    "front_high",
    "front_right",
    "right",
    "back",
    "top_down",
    "low_angle",
    "left",
]

_DEFAULT_MODELS = {
    # gpt-5.4 stopped serving image+tools requests on this account in
    # mid-May 2026 (reliable 500 from the backend, fast-fail < 1.2 s; the
    # status page never reflected an incident).  gpt-5.5 covers the same
    # capability surface and returns clean tool calls — switching as the
    # default until gpt-5.4 is healthy again.
    "openai": "gpt-5.5",
    "anthropic": "claude-sonnet-4-6",
}


class _SDKSentinelEncoder(json.JSONEncoder):
    """JSON encoder that converts SDK sentinel objects to ``null``.

    OpenAI's SDK uses ``Omit`` (and Anthropic uses ``NotGiven``) as truthy
    placeholders for "field not provided".  These sometimes get copied into
    response objects we read and then back into our message history, which
    detonates on the next request build because the SDK's own JSON encoder
    refuses to serialise them.  We can't detect every place an SDK might
    embed one, so the safest fix is to round-trip the entire payload
    through a tolerant encoder.
    """

    def default(self, o):  # noqa: D401
        type_name = type(o).__name__
        if type_name in ("Omit", "NotGiven"):
            return None
        return super().default(o)


def _scrub_sdk_sentinels(obj: Any) -> Any:
    """Round-trip *obj* through JSON, replacing SDK sentinels with ``null``.

    Slower than the structural fallback that came before, but catches
    sentinels nested inside SDK-constructed wrappers we don't otherwise
    know about.
    """
    return json.loads(json.dumps(obj, cls=_SDKSentinelEncoder))


def _safe_str(value: Any, default: str = "") -> str:
    """Return a guaranteed-string value, treating SDK sentinels as missing."""
    if isinstance(value, str):
        return value
    return default


@dataclass
class ToolCall:
    """One tool invocation requested by the model in a single turn."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """A tool handler's output, to be appended back to the conversation.

    ``text`` is the primary string payload.  ``image_paths`` are optional
    inline images — supported natively by Anthropic (emitted inside the
    ``tool_result`` content blocks); for OpenAI they are appended as a
    separate user message after the tool-role messages.
    """

    tool_use_id: str
    text: str = ""
    image_paths: list[Path] = field(default_factory=list)
    is_error: bool = False


@dataclass
class TurnResult:
    """The parsed output of one round-trip to the provider."""

    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    assistant_message: dict[str, Any]  # provider-native, for appending to history
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class UsageStats:
    """Cumulative token usage and cost tracking across VLM calls."""

    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_retries: int = 0
    call_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def record(
        self, *, input_tokens: int = 0, output_tokens: int = 0, model: str = "",
        latency: float = 0.0, retries: int = 0,
    ) -> None:
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_retries += retries
        self.call_log.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_s": round(latency, 2),
            "retries": retries,
        })

    def summary(self) -> str:
        return (
            f"{self.total_calls} calls, "
            f"{self.total_input_tokens:,} in + {self.total_output_tokens:,} out tokens, "
            f"{self.total_retries} retries"
        )


def _infer_provider_from_model(model: str) -> str | None:
    """Guess provider from the model name prefix."""
    m = model.lower()
    if m.startswith(("gpt-", "o1", "o3", "o4", "o5")):
        return "openai"
    if m.startswith("claude"):
        return "anthropic"
    return None


def _detect_provider() -> str:
    """Pick a provider based on which API key is set."""
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    raise RuntimeError(
        "No VLM API key found.  Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
    )


class VLMClient:
    """Thin, provider-agnostic wrapper for vision + tool-use queries."""

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        # Resolve provider
        if provider is None and model is not None:
            provider = _infer_provider_from_model(model)
        if provider is None:
            provider = _detect_provider()

        self.provider = provider
        self.model = model or _DEFAULT_MODELS[provider]
        self.max_retries = max_retries
        self.usage = UsageStats()

        if provider == "openai":
            import openai

            self.client = openai.OpenAI()
        elif provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic()
        else:
            raise ValueError(f"Unknown provider: {provider!r}")

        logger.info("VLM: provider=%s  model=%s", self.provider, self.model)

    # ------------------------------------------------------------------
    # Image encoding (provider-specific content blocks)
    # ------------------------------------------------------------------

    def encode_image(self, path: Path) -> dict:
        """Return a provider-appropriate image content block."""
        data = base64.standard_b64encode(path.read_bytes()).decode()
        media_type = _MEDIA_TYPES.get(path.suffix.lower().lstrip("."), "image/png")

        if self.provider == "openai":
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{data}"},
            }

        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }

    # ------------------------------------------------------------------
    # Tool-use query
    # ------------------------------------------------------------------

    def query_with_tool(
        self,
        *,
        system: str,
        user_content: list[dict],
        tool: dict,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        """Send a VLM query forcing a specific tool call.

        *tool* uses Anthropic-style schema (``name``, ``description``,
        ``input_schema``).  Converted automatically for OpenAI.

        Retries transient errors with exponential backoff.
        """
        last_err: Exception | None = None
        retries = 0

        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                if self.provider == "openai":
                    result, tok_in, tok_out = self._query_openai(
                        system, user_content, tool, max_tokens,
                    )
                else:
                    result, tok_in, tok_out = self._query_anthropic(
                        system, user_content, tool, max_tokens,
                    )
                elapsed = time.time() - t0
                self.usage.record(
                    input_tokens=tok_in, output_tokens=tok_out,
                    model=self.model, latency=elapsed, retries=retries,
                )
                logger.info(
                    "VLM call: %d in + %d out tokens, %.1fs, %d retries",
                    tok_in, tok_out, elapsed, retries,
                )
                return result

            except Exception as exc:
                last_err = exc
                if not self._is_retryable(exc) or attempt >= self.max_retries:
                    raise
                retries += 1
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "VLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, self.max_retries + 1, exc, delay,
                )
                time.sleep(delay)

        raise last_err  # unreachable, but satisfies type checker

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Check if an exception is transient and worth retrying."""
        exc_type = type(exc).__name__
        # Rate limits, server errors, timeouts, connection errors
        if exc_type in (
            "RateLimitError", "APITimeoutError", "APIConnectionError",
            "InternalServerError", "ServiceUnavailableError",
        ):
            return True
        # Check HTTP status code if available
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        if isinstance(status, int) and status in (429, 500, 502, 503, 504):
            return True
        # Connection-level errors
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return True
        return False

    # ---- OpenAI ----

    def _query_openai(
        self, system: str, user_content: list, tool: dict, max_tokens: int,
    ) -> tuple[dict[str, Any], int, int]:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["input_schema"],
            },
        }
        response = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            tools=[openai_tool],
            tool_choice={"type": "function", "function": {"name": tool["name"]}},
        )
        msg = response.choices[0].message

        # Extract token usage
        tok_in = tok_out = 0
        if response.usage:
            tok_in = response.usage.prompt_tokens or 0
            tok_out = response.usage.completion_tokens or 0

        if msg.tool_calls:
            return json.loads(msg.tool_calls[0].function.arguments), tok_in, tok_out
        raise ValueError(f"Expected tool call in response, got: {msg}")

    # ---- Anthropic ----

    def _query_anthropic(
        self, system: str, user_content: list, tool: dict, max_tokens: int,
    ) -> tuple[dict[str, Any], int, int]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_content}],
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
        )

        # Extract token usage
        tok_in = tok_out = 0
        if response.usage:
            tok_in = response.usage.input_tokens or 0
            tok_out = response.usage.output_tokens or 0

        for block in response.content:
            if block.type == "tool_use":
                return block.input, tok_in, tok_out
        raise ValueError(f"Expected tool_use in response, got: {response.content}")

    # ------------------------------------------------------------------
    # Agentic multi-tool / multi-turn
    # ------------------------------------------------------------------

    def run_turn(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 8192,
        tool_choice: str = "auto",
    ) -> TurnResult:
        """One round-trip that may emit text + zero or more tool calls.

        *messages* is the full provider-native conversation history.  The
        returned ``assistant_message`` should be appended to it verbatim
        before the next turn.

        *tools* is a list of Anthropic-style schemas (``name``,
        ``description``, ``input_schema``); converted automatically for
        OpenAI.  ``tool_choice`` supports ``"auto"`` (model decides),
        ``"any"`` (must call some tool), or ``"none"`` (text only).

        Retries on transient errors with exponential backoff.
        """
        last_err: Exception | None = None
        retries = 0

        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                if self.provider == "openai":
                    result = self._run_turn_openai(
                        system, messages, tools, max_tokens, tool_choice,
                    )
                else:
                    result = self._run_turn_anthropic(
                        system, messages, tools, max_tokens, tool_choice,
                    )
                elapsed = time.time() - t0
                self.usage.record(
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    model=self.model,
                    latency=elapsed,
                    retries=retries,
                )
                logger.info(
                    "VLM turn: %d in + %d out tokens, %d tool calls, stop=%s, %.1fs",
                    result.input_tokens, result.output_tokens,
                    len(result.tool_calls), result.stop_reason, elapsed,
                )
                return result

            except Exception as exc:
                last_err = exc
                if not self._is_retryable(exc) or attempt >= self.max_retries:
                    raise
                retries += 1
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "VLM turn failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, self.max_retries + 1, exc, delay,
                )
                time.sleep(delay)

        raise last_err  # unreachable

    def append_tool_results(
        self,
        messages: list[dict],
        results: list[ToolResult],
    ) -> None:
        """Append tool results to *messages* in provider-native format.

        Anthropic: one ``user`` message containing all ``tool_result`` blocks
        (with optional inline images per block).

        OpenAI: one ``tool`` message per result (text only), optionally
        followed by one ``user`` message bundling any images — OpenAI's
        ``tool`` role does not accept image content.
        """
        if not results:
            return

        if self.provider == "anthropic":
            content = []
            for r in results:
                blocks: list[dict] = []
                if r.text:
                    blocks.append({"type": "text", "text": r.text})
                for p in r.image_paths:
                    blocks.append(self.encode_image(p))
                if not blocks:
                    blocks = [{"type": "text", "text": "(no output)"}]
                content.append({
                    "type": "tool_result",
                    "tool_use_id": r.tool_use_id,
                    "content": blocks,
                    "is_error": r.is_error,
                })
            messages.append({"role": "user", "content": content})
            return

        # OpenAI
        any_images: list[Path] = []
        for r in results:
            body = r.text or ("(error)" if r.is_error else "(no output)")
            messages.append({
                "role": "tool",
                "tool_call_id": r.tool_use_id,
                "content": body,
            })
            any_images.extend(r.image_paths)
        if any_images:
            user_blocks: list[dict] = [
                {"type": "text", "text": "Tool-produced images:"},
            ]
            for p in any_images:
                user_blocks.append(self.encode_image(p))
            messages.append({"role": "user", "content": user_blocks})

    # -- Anthropic turn --

    def _run_turn_anthropic(
        self, system: str, messages: list[dict], tools: list[dict],
        max_tokens: int, tool_choice: str,
    ) -> TurnResult:
        tc_param: dict[str, Any]
        if tool_choice == "any":
            tc_param = {"type": "any"}
        elif tool_choice == "none":
            tc_param = {"type": "none"}
        else:
            tc_param = {"type": "auto"}

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            tool_choice=tc_param,
        )

        tok_in = response.usage.input_tokens if response.usage else 0
        tok_out = response.usage.output_tokens if response.usage else 0

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        assistant_content: list[dict] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id, name=block.name, arguments=dict(block.input),
                ))
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        return TurnResult(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "",
            assistant_message={"role": "assistant", "content": assistant_content},
            input_tokens=tok_in,
            output_tokens=tok_out,
        )

    # -- OpenAI turn --

    def _run_turn_openai(
        self, system: str, messages: list[dict], tools: list[dict],
        max_tokens: int, tool_choice: str,
    ) -> TurnResult:
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]
        tc_param: Any
        if tool_choice == "any":
            tc_param = "required"
        elif tool_choice == "none":
            tc_param = "none"
        else:
            tc_param = "auto"

        # Scrub any sentinel values that may have crept into accumulated
        # history from a prior turn before sending to the API.  Apply to
        # everything we pass — messages history is the most likely host
        # but tools/tool_choice are not immune.
        full_messages = _scrub_sdk_sentinels(
            [{"role": "system", "content": system}, *messages],
        )
        clean_tools = _scrub_sdk_sentinels(openai_tools)

        # Diagnostic: if anything still won't JSON-serialise, dump the
        # offending payload to /tmp so we can see what the SDK trips on.
        try:
            json.dumps({
                "messages": full_messages,
                "tools": clean_tools,
                "tool_choice": tc_param,
            })
        except TypeError as exc:
            import os
            dump_path = "/tmp/vlm_payload_unserialisable.json"
            try:
                with open(dump_path, "w") as f:
                    json.dump(
                        {"messages": full_messages, "tools": clean_tools,
                         "tool_choice": tc_param},
                        f, default=lambda o: f"<{type(o).__name__}>", indent=2,
                    )
                logger.error("Payload not JSON-serialisable (%s) — dumped to %s",
                             exc, dump_path)
            except Exception:
                logger.error("Payload not JSON-serialisable (%s); dump failed",
                             exc, exc_info=True)
            raise

        response = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens,
            messages=full_messages,
            tools=clean_tools,
            tool_choice=tc_param,
        )
        msg = response.choices[0].message

        tok_in = response.usage.prompt_tokens if response.usage else 0
        tok_out = response.usage.completion_tokens if response.usage else 0

        # The OpenAI SDK uses an internal ``Omit`` sentinel (not ``None``)
        # for unset fields.  Strict-coerce every external attribute we
        # store back into our message history.
        text = _safe_str(msg.content)
        raw_tool_calls = msg.tool_calls if isinstance(msg.tool_calls, list) else []

        tool_calls: list[ToolCall] = []
        assistant_tool_calls: list[dict] = []

        for tc in raw_tool_calls:
            tc_id = _safe_str(getattr(tc, "id", None))
            fn = getattr(tc, "function", None)
            tc_name = _safe_str(getattr(fn, "name", None))
            args_str = _safe_str(getattr(fn, "arguments", None), default="{}") or "{}"
            if not tc_id or not tc_name:
                logger.warning(
                    "Dropping malformed tool call from response: id=%r name=%r",
                    tc_id, tc_name,
                )
                continue
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc_id, name=tc_name, arguments=args))
            assistant_tool_calls.append({
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": tc_name,
                    "arguments": args_str,
                },
            })

        assistant_message: dict[str, Any] = {"role": "assistant", "content": text}
        if assistant_tool_calls:
            assistant_message["tool_calls"] = assistant_tool_calls

        return TurnResult(
            text=text,
            tool_calls=tool_calls,
            stop_reason=response.choices[0].finish_reason or "",
            assistant_message=assistant_message,
            input_tokens=tok_in,
            output_tokens=tok_out,
        )

    # ------------------------------------------------------------------
    # Image selection helpers
    # ------------------------------------------------------------------

    def select_images(
        self, image_paths: list[Path], max_count: int = MAX_IMAGES_PER_CALL,
    ) -> list[Path]:
        """Pick a diverse subset of images if there are too many."""
        if len(image_paths) <= max_count:
            return list(image_paths)

        selected: list[Path] = []
        remaining = list(image_paths)

        for name in _PRIORITY_VIEWS:
            for p in remaining:
                if p.stem == name:
                    selected.append(p)
                    remaining.remove(p)
                    break
            if len(selected) >= max_count:
                break

        while len(selected) < max_count and remaining:
            selected.append(remaining.pop(0))

        return selected
