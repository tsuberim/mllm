"""
Tests for harness/agent.py using a mock OpenAI client.

The mock intercepts client.chat.completions.create() and returns pre-scripted
responses, while the real Sandbox executes bash commands.
"""

import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch
from pathlib import Path

from harness.agent import run_agent, AgentConfig
from harness.protocol import (
    T_THINK_O, T_THINK_C, T_CALL_O, T_CALL_C, T_DONE_O, T_DONE_C,
    format_task,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeChoice:
    message: object
    finish_reason: str
    stop_reason: str = None

    @property
    def text(self):
        return self.message.content or ""


@dataclass
class FakeMessage:
    content: str


@dataclass
class FakeResponse:
    choices: list


def make_response(content: str, finish_reason: str = "stop") -> FakeResponse:
    """Build a fake response. finish_reason should be the stop token that triggered
    the stop (e.g. T_CALL_C), or 'stop'/'length' for natural stops."""
    return FakeResponse(choices=[FakeChoice(FakeMessage(content), finish_reason, stop_reason=finish_reason)])


def mock_client(responses: list[FakeResponse]):
    """Return a mock OpenAI client that serves responses in order."""
    client = MagicMock()
    client.chat.completions.create.side_effect = responses
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return AgentConfig(
        vllm_url="http://localhost:8000/v1",
        model="test-model",
        max_new_tokens=512,
        max_trace_tokens=8192,
        temperature=0.0,
        bash_timeout=10,
        trace_timeout=30,
    )


@pytest.fixture
def sandbox(tmp_path):
    """Minimal sandbox stub — runs bash locally for tests."""
    from harness.sandbox import BashResult
    import subprocess

    sb = MagicMock()
    def _bash(cmd, timeout=30):
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return BashResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
        )
    sb.bash.side_effect = _bash
    return sb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_simple_done_no_tool_calls(config, sandbox):
    """Agent that immediately says done without any tool calls."""
    responses = [
        make_response(f"{T_THINK_O}nothing to do{T_THINK_C}{T_DONE_O}42{T_DONE_C}"),
    ]
    with patch("harness.agent.OpenAI", return_value=mock_client(responses)):
        result = run_agent("What is 6*7?", sandbox, config)

    assert result.success is True
    assert result.answer == "42"
    assert result.n_tool_calls == 0


def test_one_tool_call_then_done(config, sandbox):
    """Agent makes one bash call, gets result, then finishes."""
    responses = [
        make_response(f"{T_THINK_O}check files{T_THINK_C}{T_CALL_O}bash echo hello{T_CALL_C}", T_CALL_C),
        make_response(f"{T_THINK_O}got it{T_THINK_C}{T_DONE_O}done{T_DONE_C}", T_DONE_C),
    ]
    with patch("harness.agent.OpenAI", return_value=mock_client(responses)):
        result = run_agent("Say hello", sandbox, config)

    assert result.success is True
    assert result.n_tool_calls == 1
    assert "hello" in result.trace


def test_multiple_tool_calls(config, sandbox):
    """Agent chains three bash commands before finishing."""
    responses = [
        make_response(f"{T_CALL_O}bash echo step1{T_CALL_C}", T_CALL_C),
        make_response(f"{T_CALL_O}bash echo step2{T_CALL_C}", T_CALL_C),
        make_response(f"{T_CALL_O}bash echo step3{T_CALL_C}", T_CALL_C),
        make_response(f"{T_DONE_O}{T_DONE_C}", T_DONE_C),
    ]
    with patch("harness.agent.OpenAI", return_value=mock_client(responses)):
        result = run_agent("Do three things", sandbox, config)

    assert result.success is True
    assert result.n_tool_calls == 3


def test_validator_pass(config, sandbox):
    """Validator returning True marks trace as success."""
    responses = [make_response(f"{T_DONE_O}{T_DONE_C}")]
    validator = lambda sb, ans: True

    with patch("harness.agent.OpenAI", return_value=mock_client(responses)):
        result = run_agent("task", sandbox, config, validator=validator)

    assert result.success is True


def test_validator_fail(config, sandbox):
    """Validator returning False marks trace as failed."""
    responses = [make_response(f"{T_DONE_O}{T_DONE_C}")]
    validator = lambda sb, ans: False

    with patch("harness.agent.OpenAI", return_value=mock_client(responses)):
        result = run_agent("task", sandbox, config, validator=validator)

    assert result.success is False
    assert result.failure_reason is None  # failed by validator, not by error


def test_validator_exception_treated_as_failure(config, sandbox):
    """If validator raises, treat as failure."""
    responses = [make_response(f"{T_DONE_O}{T_DONE_C}")]
    def bad_validator(sb, ans):
        raise RuntimeError("validator crashed")

    with patch("harness.agent.OpenAI", return_value=mock_client(responses)):
        result = run_agent("task", sandbox, config, validator=bad_validator)

    assert result.success is False


def test_context_budget_exceeded(config, sandbox):
    """Agent fails when context exceeds max_trace_tokens after a tool call."""
    # Make a budget that fits the initial context but not after a tool result
    small_config = AgentConfig(
        vllm_url="http://localhost:8000/v1",
        model="test-model",
        max_trace_tokens=20,  # fits task+one-call but not tool result + second turn
        bash_timeout=10,
        trace_timeout=30,
    )
    # First call succeeds, then context is over budget
    responses = [
        make_response(f"{T_CALL_O}bash echo {'x' * 500}{T_CALL_C}", T_CALL_C),
        make_response(f"{T_DONE_O}{T_DONE_C}", T_DONE_C),
    ]

    with patch("harness.agent.OpenAI", return_value=mock_client(responses)):
        result = run_agent("t", sandbox, small_config)

    assert result.success is False
    assert result.failure_reason == "context budget exceeded"


def test_unexpected_stop_reason(config, sandbox):
    """Model stops with finish_reason='stop' and no recognized token → failure."""
    responses = [make_response("some rambling without protocol tokens", "stop")]

    with patch("harness.agent.OpenAI", return_value=mock_client(responses)):
        result = run_agent("task", sandbox, config)

    assert result.success is False
    assert "unexpected stop" in result.failure_reason


def test_trace_contains_tool_results(config, sandbox):
    """Tool results from bash are embedded in the trace."""
    responses = [
        make_response(f"{T_CALL_O}bash echo MARKER{T_CALL_C}"),
        make_response(f"{T_DONE_O}{T_DONE_C}"),
    ]
    with patch("harness.agent.OpenAI", return_value=mock_client(responses)):
        result = run_agent("task", sandbox, config)

    assert "MARKER" in result.trace


def test_strip_thinking_in_trace_sft(config, sandbox):
    """trace_sft (thinking-stripped) is produced correctly by the runner, not agent."""
    from harness.protocol import strip_thinking
    responses = [
        make_response(f"{T_THINK_O}secret reasoning{T_THINK_C}{T_DONE_O}answer{T_DONE_C}"),
    ]
    with patch("harness.agent.OpenAI", return_value=mock_client(responses)):
        result = run_agent("task", sandbox, config)

    sft = strip_thinking(result.trace)
    assert "secret reasoning" not in sft
    assert "answer" in sft


def test_done_answer_parsed(config, sandbox):
    """Answer text inside <|done|>...<|/done|> is returned correctly."""
    responses = [make_response(f"{T_DONE_O}The answer is 99{T_DONE_C}")]

    with patch("harness.agent.OpenAI", return_value=mock_client(responses)):
        result = run_agent("task", sandbox, config)

    assert result.answer == "The answer is 99"


def test_bare_done_token(config, sandbox):
    """<|done|> without closing tag is handled gracefully."""
    responses = [make_response(f"{T_DONE_O}")]

    with patch("harness.agent.OpenAI", return_value=mock_client(responses)):
        result = run_agent("task", sandbox, config)

    assert result.success is True
    assert T_DONE_C in result.trace  # auto-closed
