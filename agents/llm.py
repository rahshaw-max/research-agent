"""Shared Anthropic client + tool-forced structured output helper."""
from __future__ import annotations

from typing import Type, TypeVar

import anthropic
import httpx2 as httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()

from agents.config import MODEL_FAST, MODEL_SYNTHESIS  # re-export

# Explicit timeouts and SDK-level retry budget (handles 429 / 5xx automatically).
# Application-level retries in researcher.py add on top of this for the inner loop.
client = anthropic.AsyncAnthropic(
    max_retries=4,
    timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
)

T = TypeVar("T", bound=BaseModel)


async def tool_forced_call(
    *,
    model: str,
    max_tokens: int,
    system: str | list,
    messages: list,
    tool: dict,
    schema: Type[T],
    retries: int = 2,
    _usage: dict | None = None,
) -> T:
    """
    Call the model with tool-forced output, validate the input with Pydantic,
    and retry (up to `retries` attempts) feeding the validation error back.
    """
    tool_name = tool["name"]
    msgs = list(messages)

    for attempt in range(retries):
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=msgs,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool_name},
        )

        if _usage is not None:
            _usage["input_tokens"]         = _usage.get("input_tokens",         0) + response.usage.input_tokens
            _usage["output_tokens"]        = _usage.get("output_tokens",        0) + response.usage.output_tokens
            _usage["cache_read_tokens"]    = _usage.get("cache_read_tokens",    0) + getattr(response.usage, "cache_read_input_tokens",    0)
            _usage["cache_created_tokens"] = _usage.get("cache_created_tokens", 0) + getattr(response.usage, "cache_creation_input_tokens", 0)

        tool_use_block = None
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
                tool_use_block = block
                break

        if tool_use_block is None:
            raise ValueError(f"Model did not call tool '{tool_name}'")

        try:
            return schema(**tool_use_block.input)
        except ValidationError as exc:
            if attempt < retries - 1:
                msgs = msgs + [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": tool_use_block.id,
                                "name": tool_use_block.name,
                                "input": tool_use_block.input,
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_block.id,
                                "is_error": True,
                                "content": f"Validation error: {exc}\nFix the issues and resubmit.",
                            }
                        ],
                    },
                ]
            else:
                raise ValueError(f"tool_forced_call failed after {retries} attempts: {exc}") from exc

    raise RuntimeError("Unreachable")
