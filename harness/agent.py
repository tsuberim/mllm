"""
Single agent execution loop.

Given a task and a sandbox, drives the model via vLLM until <|done|>.
Returns the full raw trace (including thinking) and metadata.

Spawn/wait are parsed but not yet implemented — leaf tasks only for now.
"""

import time
from dataclasses import dataclass, field
from openai import OpenAI

from .protocol import (
    T_CALL_O, T_CALL_C, T_SPAWN_O, T_SPAWN_C,
    T_WAIT_O, T_WAIT_C, T_DONE_O, T_DONE_C,
    STOP_SEQUENCES,
    format_task, format_tool_result, format_agent_id, format_wait_result,
    parse_tool_call, parse_spawn, parse_wait, parse_done,
)
from .sandbox import Sandbox, TRACE_TIMEOUT_S

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Merlin, a coding agent. You work inside a Python 3.12 sandbox with standard Unix tools available (bash, grep, rg, find, fd, git, curl, jq, yq, python3, pytest, etc.).

You receive a task and solve it by running bash commands. Follow this exact format:

<|think|>
Brief reasoning — what you need to do and how. Keep it short.
<|/think|>
<|tool_call|>bash <command><|/tool_call|>

The result will be injected as:
<|tool_result|>...<|/tool_result|>

Repeat as needed. When done:
<|done|>brief answer if needed<|/done|>
or just:
<|done|>

Rules:
- Always use <|tool_call|>bash ...<|/tool_call|> format exactly — no markdown, no code blocks
- Keep thinking brief (1-3 lines)
- Use standard Unix commands — grep, rg, find, sed, cat, python3, pytest, etc.
- If a command output is truncated, page through with head/tail/sed
- Check exit codes — non-zero means something went wrong
- For fix-the-bug tasks: run pytest first to see failures, read the failing code, fix it, verify with pytest
"""


@dataclass
class AgentConfig:
    vllm_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    max_new_tokens: int = 512   # per generation step
    max_trace_tokens: int = 4096  # total context budget (leave room for task + results)
    temperature: float = 0.7
    top_p: float = 0.95
    bash_timeout: int = 30
    trace_timeout: int = TRACE_TIMEOUT_S


@dataclass
class TraceResult:
    task: str
    repo: str
    trace: str                  # full raw trace including thinking
    success: bool
    answer: str | None
    n_tool_calls: int
    n_tokens: int               # approximate
    duration_s: float
    failure_reason: str | None = None


def run_agent(
    task: str,
    sandbox: Sandbox,
    config: AgentConfig,
    repo_name: str = "",
    validator=None,             # optional callable(sandbox, answer) -> bool
) -> TraceResult:
    """
    Run a single agent loop: task → tool calls → done.
    Returns TraceResult with full trace and outcome.
    """
    client = OpenAI(base_url=config.vllm_url, api_key="none")

    context = format_task(task)
    n_tool_calls = 0
    start = time.monotonic()

    def _elapsed():
        return time.monotonic() - start

    def _approx_tokens(text: str) -> int:
        return len(text) // 4  # rough chars-per-token estimate

    try:
        while _elapsed() < config.trace_timeout:
            if _approx_tokens(context) >= config.max_trace_tokens:
                return _fail(task, repo_name, context, n_tool_calls, _elapsed(),
                             "context budget exceeded")

            # Generate next chunk — stops at a tool call, spawn, wait, or done
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stop=STOP_SEQUENCES,
            )
            chunk = response.choices[0].message.content or ""
            stop_reason = response.choices[0].finish_reason
            stop_token = _extract_stop_token(response)

            context += chunk

            # --- tool call ---
            if stop_token == T_CALL_C or T_CALL_O in chunk:
                # Extract command between <|tool_call|> and <|/tool_call|>.
                # T_CALL_C may or may not already be in context depending on
                # whether vLLM includes the stop token in the output.
                cmd_text = _extract_between(context, T_CALL_O, T_CALL_C)
                if cmd_text is None:
                    return _fail(task, repo_name, context, n_tool_calls, _elapsed(),
                                 "malformed tool call")

                if T_CALL_C not in context:
                    context += T_CALL_C
                context += "\n"
                cmd = parse_tool_call(cmd_text)
                result = sandbox.bash(cmd, timeout=config.bash_timeout)
                context += format_tool_result(result.stdout, result.stderr, result.exit_code)
                n_tool_calls += 1

            # --- spawn (not yet implemented) ---
            elif stop_token == T_SPAWN_C:
                return _fail(task, repo_name, context, n_tool_calls, _elapsed(),
                             "spawn not yet supported in leaf agent")

            # --- wait (not yet implemented) ---
            elif stop_token == T_WAIT_C:
                return _fail(task, repo_name, context, n_tool_calls, _elapsed(),
                             "wait not yet supported in leaf agent")

            # --- done ---
            elif stop_token in (T_DONE_O, T_DONE_C) or T_DONE_O in chunk:
                answer_text = _extract_between(context, T_DONE_O, T_DONE_C)
                answer = parse_done(answer_text or "")

                # Close bare <|done|> if needed
                if T_DONE_C not in context:
                    context += T_DONE_C

                # Run validator if provided
                success = True
                if validator is not None:
                    try:
                        success = validator(sandbox, answer)
                    except Exception:
                        success = False

                return TraceResult(
                    task=task,
                    repo=repo_name,
                    trace=context,
                    success=success,
                    answer=answer,
                    n_tool_calls=n_tool_calls,
                    n_tokens=_approx_tokens(context),
                    duration_s=_elapsed(),
                )

            elif stop_reason == "stop" or stop_reason == "length":
                # Model stopped without a recognized token — treat as failure
                return _fail(task, repo_name, context, n_tool_calls, _elapsed(),
                             f"unexpected stop: {stop_reason}")

        return _fail(task, repo_name, context, n_tool_calls, _elapsed(), "trace timeout")

    except Exception as e:
        return _fail(task, repo_name, context, n_tool_calls, _elapsed(), str(e))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail(task, repo, trace, n_calls, duration, reason) -> TraceResult:
    return TraceResult(
        task=task, repo=repo, trace=trace, success=False, answer=None,
        n_tool_calls=n_calls, n_tokens=len(trace) // 4,
        duration_s=duration, failure_reason=reason,
    )


def _extract_between(text: str, open_tok: str, close_tok: str | None) -> str | None:
    """Extract content between the last open_tok and close_tok (or end of string)."""
    idx = text.rfind(open_tok)
    if idx == -1:
        return None
    start = idx + len(open_tok)
    if close_tok is None:
        return text[start:]
    end = text.find(close_tok, start)
    if end == -1:
        return text[start:]
    return text[start:end]


def _extract_stop_token(response) -> str | None:
    """Try to recover which stop sequence triggered the stop."""
    # vLLM includes stop_reason in the response when a stop sequence is hit
    choice = response.choices[0]
    # Check finish_reason — vLLM sets it to the matched stop string when applicable
    reason = getattr(choice, "stop_reason", None) or choice.finish_reason
    if reason in STOP_SEQUENCES:
        return reason
    # Fallback: infer from text content
    text = choice.text
    for tok in STOP_SEQUENCES:
        if text.rstrip().endswith(tok.rstrip(">")):
            return tok
    return None
