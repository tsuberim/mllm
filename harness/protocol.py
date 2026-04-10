"""
Protocol token definitions and trace formatting helpers.

Special tokens (must match tokenizer special tokens):
  <|task|>         <|/task|>       — task instruction
  <|think|>        <|/think|>      — model reasoning (stripped before SFT)
  <|tool_call|>    <|/tool_call|>  — bash command to execute
  <|tool_result|>  <|/tool_result|>— result injected by harness
  <|spawn|>        <|/spawn|>      — spawn a child agent (non-blocking)
  <|agent_id|>     <|/agent_id|>   — returned agent id from spawn
  <|wait|>         <|/wait|>       — wait for agent ids to complete
  <|wait_result|>  <|/wait_result|>— completion status per agent
  <|done|>         <|/done|>       — task complete, optional answer inline
"""

import re

# ---------------------------------------------------------------------------
# Token constants
# ---------------------------------------------------------------------------

T_TASK_O       = "<|task|>"
T_TASK_C       = "<|/task|>"
T_THINK_O      = "<|think|>"
T_THINK_C      = "<|/think|>"
T_CALL_O       = "<|tool_call|>"
T_CALL_C       = "<|/tool_call|>"
T_RESULT_O     = "<|tool_result|>"
T_RESULT_C     = "<|/tool_result|>"
T_SPAWN_O      = "<|spawn|>"
T_SPAWN_C      = "<|/spawn|>"
T_AGENT_ID_O   = "<|agent_id|>"
T_AGENT_ID_C   = "<|/agent_id|>"
T_WAIT_O       = "<|wait|>"
T_WAIT_C       = "<|/wait|>"
T_WAIT_RES_O   = "<|wait_result|>"
T_WAIT_RES_C   = "<|/wait_result|>"
T_DONE_O       = "<|done|>"
T_DONE_C       = "<|/done|>"

ALL_SPECIAL_TOKENS = [
    T_TASK_O, T_TASK_C,
    T_THINK_O, T_THINK_C,
    T_CALL_O, T_CALL_C,
    T_RESULT_O, T_RESULT_C,
    T_SPAWN_O, T_SPAWN_C,
    T_AGENT_ID_O, T_AGENT_ID_C,
    T_WAIT_O, T_WAIT_C,
    T_WAIT_RES_O, T_WAIT_RES_C,
    T_DONE_O, T_DONE_C,
]

# Stop sequences for vLLM — model stops generating at any of these
STOP_SEQUENCES = [T_CALL_C, T_SPAWN_C, T_WAIT_C, T_DONE_O, T_DONE_C]

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_task(instruction: str) -> str:
    """Opening of a trace."""
    return f"{T_TASK_O}{instruction}{T_TASK_C}\n"


def format_tool_result(stdout: str, stderr: str, exit_code: int) -> str:
    """Wrap harness-injected bash result."""
    content = stdout
    if stderr:
        content += stderr if not stdout else f"\n{stderr}"
    if exit_code != 0:
        content += f"\n# exit code: {exit_code}"
    return f"{T_RESULT_O}{content}{T_RESULT_C}\n"


def format_wait_result(statuses: dict[str, str]) -> str:
    """Format wait result block. statuses: {agent_id: 'completed'|'failed'}"""
    lines = "\n".join(f"{aid} {status}" for aid, status in statuses.items())
    return f"{T_WAIT_RES_O}\n{lines}\n{T_WAIT_RES_C}\n"


def format_agent_id(agent_id: str) -> str:
    return f"{T_AGENT_ID_O}{agent_id}{T_AGENT_ID_C}\n"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_tool_call(text: str) -> str | None:
    """Extract bash command from a tool_call block (open tag already consumed)."""
    # text is everything after <|tool_call|>, up to (not including) <|/tool_call|>
    return text.strip()


def parse_spawn(text: str) -> tuple[str, list[str]]:
    """
    Parse spawn content: 'task -- file1 file2'
    Returns (task, context_files).
    """
    if " -- " in text:
        task, files_str = text.split(" -- ", 1)
        context_files = files_str.split()
    else:
        task = text
        context_files = []
    return task.strip(), context_files


def parse_wait(text: str) -> list[str]:
    """Parse wait content: space-separated agent ids."""
    return text.strip().split()


def parse_done(text: str) -> str | None:
    """
    Parse done token content.
    Returns answer string if present, None if bare <|done|>.
    """
    return text.strip() if text.strip() else None


def strip_thinking(trace: str) -> str:
    """Remove all <|think|>...</|think|> blocks from a trace (for SFT corpus)."""
    return re.sub(
        re.escape(T_THINK_O) + r".*?" + re.escape(T_THINK_C),
        "",
        trace,
        flags=re.DOTALL,
    ).strip()
