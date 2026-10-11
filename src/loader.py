"""Robust JSON loader and data matcher for the LLM evaluation pipeline.

Handles messy input data: invalid JSON syntax (C-style comments, trailing
commas), mismatched records, and inconsistent response formats.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# Type aliases for clarity
EvalRecord = dict[str, str]
LogMessages = list[str]
ParseResult = tuple[Any | None, str | None, bool]


def clean_and_parse_json(file_path: str) -> ParseResult:
    """Read a file, auto-repair common JSON issues, and return parsed data.

    Returns:
        A 3-tuple of (parsed_data, warning_message, was_repaired).
        On failure, parsed_data is None and warning_message contains the error.
    """
    file_name = os.path.basename(file_path)

    # Fast path: try standard parse first
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f), None, False
    except json.JSONDecodeError:
        pass

    # Slow path: strip C-style line comments and trailing commas
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()

        # Remove single-line comments (// ...) that are NOT inside JSON strings.
        # We use a regex that matches quoted strings first (to skip them) and
        # then matches bare // comments outside of strings.
        raw = re.sub(
            r'"(?:[^"\\]|\\.)*"'  # match a quoted string (skip it)
            r"|"
            r"(//[^\n]*)",        # or match a line comment (capture it)
            lambda m: "" if m.group(1) is not None else m.group(0),
            raw,
        )

        # Remove trailing commas before ] or }
        raw = re.sub(r",\s*([}\]])", r"\1", raw)

        data = json.loads(raw)
        msg = f"⚠️ **Format Warning ({file_name}):** Auto-repaired comments/trailing commas."
        return data, msg, True

    except (json.JSONDecodeError, OSError) as exc:
        return None, f"❌ **Error ({file_name}):** {exc!s}", False


def load_data(chat_path: str, context_path: str) -> tuple[list[EvalRecord], LogMessages]:
    """Load and merge chat history with context data for evaluation.

    Returns:
        A 2-tuple of (dataset, logs) where dataset is a list of evaluation
        records and logs is a list of human-readable diagnostic messages.
    """
    logs: LogMessages = []

    # --- 1. File existence validation ---
    for path in (chat_path, context_path):
        if not os.path.exists(path):
            return [], [f"❌ Error: {path} not found"]

    chat_data, chat_msg, _ = clean_and_parse_json(chat_path)
    ctx_data, ctx_msg, _ = clean_and_parse_json(context_path)

    if chat_msg:
        logs.append(chat_msg)
    if ctx_msg:
        logs.append(ctx_msg)
    if not chat_data or not ctx_data:
        return [], logs

    try:
        # --- 2. Extract context chunks ---
        vector_data = ctx_data.get("data", {}).get("vector_data", [])
        if not vector_data:
            vector_data = ctx_data.get("vector_data", [])

        context_text = "\n\n".join(
            v.get("text", "") for v in vector_data if v.get("text")
        )

        # --- 3. Extract target response (source of truth) ---
        sources = ctx_data.get("data", {}).get("sources", {})
        target_res_raw = sources.get("final_response", "")

        if isinstance(target_res_raw, list):
            target_response_str = " ".join(str(x) for x in target_res_raw)
        else:
            target_response_str = str(target_res_raw)

        # --- 4. Find matching AI turn in chat history ---
        user_query = "Unknown Query"
        found_response: str | None = None

        turns: list[dict[str, Any]] = chat_data.get("conversation_turns", [])

        # Match the AI turn whose message overlaps with the context response
        min_match_len = 10
        match_prefix_len = 30
        for i, turn in enumerate(turns):
            if turn.get("role") != "AI/Chatbot":
                continue
            msg = turn.get("message", "")
            if (
                len(target_response_str) > min_match_len
                and (
                    target_response_str[:match_prefix_len] in msg
                    or msg[:match_prefix_len] in target_response_str
                )
            ):
                found_response = msg
                if i > 0:
                    user_query = turns[i - 1].get("message", user_query)
                break

        # --- 5. Fallback: AI response missing from chat log ---
        if not found_response:
            if turns:
                # Walk backwards to find the last user message
                for i in range(len(turns) - 1, -1, -1):
                    if turns[i].get("role") == "User":
                        user_query = turns[i].get("message", user_query)
                        break

                found_response = target_response_str
                logs.append(
                    "ℹ️ **Note:** AI response missing from Chat Log. "
                    "Extracted directly from Vector Data."
                )
            else:
                logs.append("❌ Error: Chat log is empty.")
                return [], logs

        return [
            {
                "query": user_query,
                "response": found_response,
                "context": context_text,
            }
        ], logs

    except (KeyError, TypeError, IndexError) as exc:
        logs.append(f"❌ **Processing Error:** {exc!s}")
        return [], logs