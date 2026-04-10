"""Tests for protocol token formatting and parsing."""

import pytest
from harness.protocol import (
    format_task, format_tool_result, format_wait_result, format_agent_id,
    parse_tool_call, parse_spawn, parse_wait, parse_done,
    strip_thinking,
    T_TASK_O, T_TASK_C, T_THINK_O, T_THINK_C, T_CALL_O, T_CALL_C,
    T_RESULT_O, T_RESULT_C, T_DONE_O, T_DONE_C,
    ALL_SPECIAL_TOKENS,
)


def test_format_task():
    out = format_task("find all TODO comments")
    assert out == "<|task|>find all TODO comments<|/task|>\n"


def test_format_tool_result_stdout_only():
    out = format_tool_result("hello\n", "", 0)
    assert T_RESULT_O in out and T_RESULT_C in out
    assert "hello" in out
    assert "exit code" not in out


def test_format_tool_result_with_stderr():
    out = format_tool_result("", "error: not found", 1)
    assert "error: not found" in out
    assert "exit code: 1" in out


def test_format_tool_result_nonzero_exit():
    out = format_tool_result("partial output", "", 2)
    assert "exit code: 2" in out


def test_format_wait_result():
    out = format_wait_result({"123": "completed", "456": "failed"})
    assert "123 completed" in out
    assert "456 failed" in out


def test_format_agent_id():
    out = format_agent_id("abc123")
    assert out == "<|agent_id|>abc123<|/agent_id|>\n"


def test_parse_tool_call():
    assert parse_tool_call("bash grep -rn 'TODO' src/") == "bash grep -rn 'TODO' src/"
    assert parse_tool_call("  ls -la  ") == "ls -la"


def test_parse_spawn_with_context():
    task, files = parse_spawn("find TODOs -- src/main.py src/utils.py")
    assert task == "find TODOs"
    assert files == ["src/main.py", "src/utils.py"]


def test_parse_spawn_no_context():
    task, files = parse_spawn("find all Python files")
    assert task == "find all Python files"
    assert files == []


def test_parse_wait():
    ids = parse_wait("123 456 789")
    assert ids == ["123", "456", "789"]


def test_parse_done_with_answer():
    assert parse_done("found 3 TODOs") == "found 3 TODOs"


def test_parse_done_empty():
    assert parse_done("") is None
    assert parse_done("   ") is None


def test_strip_thinking_removes_blocks():
    trace = (
        "<|task|>find TODOs<|/task|>\n"
        "<|think|>\nI'll use grep.\n<|/think|>\n"
        "<|tool_call|>bash grep -rn TODO src/<|/tool_call|>\n"
        "<|tool_result|>src/a.py:1: TODO<|/tool_result|>\n"
        "<|think|>\nDone.\n<|/think|>\n"
        "<|done|>1 TODO<|/done|>"
    )
    stripped = strip_thinking(trace)
    assert "<|think|>" not in stripped
    assert "<|/think|>" not in stripped
    assert "grep" in stripped
    assert "1 TODO" in stripped


def test_strip_thinking_no_thinking():
    trace = "<|task|>foo<|/task|>\n<|done|>"
    assert strip_thinking(trace) == trace


def test_all_special_tokens_count():
    assert len(ALL_SPECIAL_TOKENS) == 18  # protocol tokens only (no bos/eos)


def test_all_special_tokens_paired():
    """Every open token should have a matching close token."""
    opens  = [t for t in ALL_SPECIAL_TOKENS if not t.startswith("<|/")]
    closes = [t for t in ALL_SPECIAL_TOKENS if t.startswith("<|/")]
    assert len(opens) == len(closes)
    for o in opens:
        expected_close = o.replace("<|", "<|/")
        assert expected_close in closes, f"Missing close for {o}"
