"""Tests for src.loader — JSON parsing, comment stripping, and data matching."""

from __future__ import annotations

import json
import textwrap

import pytest

from src.loader import clean_and_parse_json, load_data

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_json(tmp_path):
    """Helper: write text to a temp file and return the path."""

    def _write(filename: str, content: str) -> str:
        p = tmp_path / filename
        p.write_text(content, encoding="utf-8")
        return str(p)

    return _write


# ── clean_and_parse_json ─────────────────────────────────────────────


class TestCleanAndParseJson:
    def test_valid_json(self, tmp_json):
        path = tmp_json("ok.json", '{"a": 1}')
        data, msg, repaired = clean_and_parse_json(path)
        assert data == {"a": 1}
        assert msg is None
        assert repaired is False

    def test_strips_line_comments(self, tmp_json):
        content = textwrap.dedent("""\
            {
                // this is a comment
                "key": "value"
            }
        """)
        path = tmp_json("commented.json", content)
        data, msg, repaired = clean_and_parse_json(path)
        assert data == {"key": "value"}
        assert repaired is True
        assert "Auto-repaired" in msg

    def test_preserves_urls_in_strings(self, tmp_json):
        """Regression: naive split('//') would break URLs inside JSON strings."""
        content = textwrap.dedent("""\
            {
                // A comment
                "url": "https://example.com/path"
            }
        """)
        path = tmp_json("url.json", content)
        data, _msg, _repaired = clean_and_parse_json(path)
        assert data["url"] == "https://example.com/path"

    def test_removes_trailing_commas(self, tmp_json):
        content = '{"a": 1, "b": [1, 2, 3,],}'
        path = tmp_json("trailing.json", content)
        data, _msg, _repaired = clean_and_parse_json(path)
        assert data == {"a": 1, "b": [1, 2, 3]}

    def test_returns_none_on_unrecoverable_error(self, tmp_json):
        path = tmp_json("broken.json", "{{not json at all}}")
        data, msg, repaired = clean_and_parse_json(path)
        assert data is None
        assert "Error" in msg
        assert repaired is False


# ── load_data ────────────────────────────────────────────────────────


def _make_chat_file(tmp_json, turns: list[dict]) -> str:
    return tmp_json("chat.json", json.dumps({"conversation_turns": turns}))


def _make_ctx_file(
    tmp_json,
    vector_texts: list[str],
    final_response: str = "",
) -> str:
    ctx = {
        "data": {
            "vector_data": [{"id": i, "text": t} for i, t in enumerate(vector_texts)],
            "sources": {"final_response": final_response},
        }
    }
    return tmp_json("ctx.json", json.dumps(ctx))


class TestLoadData:
    def test_missing_chat_file(self, tmp_json):
        ctx = _make_ctx_file(tmp_json, ["text"])
        dataset, logs = load_data("nonexistent.json", ctx)
        assert dataset == []
        assert any("not found" in log for log in logs)

    def test_missing_context_file(self, tmp_json):
        chat = _make_chat_file(tmp_json, [])
        dataset, logs = load_data(chat, "nonexistent.json")
        assert dataset == []
        assert any("not found" in log for log in logs)

    def test_empty_chat_log(self, tmp_json):
        chat = _make_chat_file(tmp_json, [])
        ctx = _make_ctx_file(tmp_json, ["context text"], final_response="answer")
        dataset, logs = load_data(chat, ctx)
        assert dataset == []
        assert any("empty" in log.lower() for log in logs)

    def test_matched_response_from_chat(self, tmp_json):
        ai_msg = "The answer is that our rooms cost Rs 800 per night."
        turns = [
            {"turn": 1, "role": "User", "message": "How much do rooms cost?"},
            {"turn": 2, "role": "AI/Chatbot", "message": ai_msg},
        ]
        chat = _make_chat_file(tmp_json, turns)
        ctx = _make_ctx_file(
            tmp_json,
            ["Rooms cost Rs 800 per night."],
            final_response=ai_msg,
        )
        dataset, _logs = load_data(chat, ctx)
        assert len(dataset) == 1
        assert dataset[0]["query"] == "How much do rooms cost?"
        assert dataset[0]["response"] == ai_msg
        assert "Rs 800" in dataset[0]["context"]

    def test_fallback_when_ai_turn_missing(self, tmp_json):
        turns = [
            {"turn": 1, "role": "User", "message": "What is the cost?"},
        ]
        chat = _make_chat_file(tmp_json, turns)
        ctx = _make_ctx_file(
            tmp_json,
            ["Cost is Rs 3,00,000."],
            final_response="A cycle costs Rs 3,00,000.",
        )
        dataset, logs = load_data(chat, ctx)
        assert len(dataset) == 1
        assert dataset[0]["query"] == "What is the cost?"
        assert dataset[0]["response"] == "A cycle costs Rs 3,00,000."
        assert any("missing from Chat Log" in log for log in logs)

    def test_list_final_response_normalised_to_string(self, tmp_json):
        turns = [
            {"turn": 1, "role": "User", "message": "Tell me more."},
        ]
        chat = _make_chat_file(tmp_json, turns)
        ctx = _make_ctx_file(
            tmp_json,
            ["Some context."],
            final_response=["part one", "part two"],
        )
        dataset, _logs = load_data(chat, ctx)
        assert len(dataset) == 1
        assert dataset[0]["response"] == "part one part two"

    def test_return_shape_is_always_tuple(self, tmp_json):
        """Regression: main.py previously assumed load_data returned a list."""
        chat = _make_chat_file(tmp_json, [])
        ctx = _make_ctx_file(tmp_json, ["text"])
        result = load_data(chat, ctx)
        assert isinstance(result, tuple)
        assert len(result) == 2
