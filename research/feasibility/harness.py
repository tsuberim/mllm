"""
Feasibility harness: validates that a 3B coding model under 4K context can
reliably execute the Merlin task set. Measures correctness, TPS, memory, and
token usage per task.

Usage:
    python harness.py                          # run all tasks
    python harness.py --task find_py_files     # single task
    python harness.py --category search        # by category
    python harness.py --model <hf-or-mlx-id>  # override model
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Optional

import psutil

try:
    from mlx_lm import load, generate
    import mlx.core as mx
except ImportError:
    print("mlx_lm not found. Install: pip install mlx-lm")
    sys.exit(1)

from tasks import TASKS, Task

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
MAX_CONTEXT_TOKENS = 9500  # 8K Merlin tokens + ~17% tokenizer efficiency gain from domain-specialized vocab
MAX_TOKENS_PER_STEP = 512
MAX_TURNS = 10

SYSTEM_PROMPT = (
    "You are a coding agent with bash access. Current directory is the project root.\n\n"
    "Run commands in ```bash blocks. Rules:\n"
    "- Use python3, not python\n"
    "- Use grep without -P flag (macOS BSD grep); use -E for extended regex\n"
    "- To read a file: cat filename\n"
    "- To write a file: cat > file.py << 'EOF'\\n<code>\\nEOF\n"
    "- If a command fails, try a simpler alternative\n"
    "- State final answer in plain text when done"
)

TOOL_CALL_RE = re.compile(r"```bash\s*\n(.*?)```", re.DOTALL)

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    task: str
    category: str
    parallelizable: bool
    success: bool
    tokens_used: int
    tps: float
    turns: int
    context_headroom: int   # MAX_CONTEXT_TOKENS - tokens_used
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


def execute_bash(command: str, cwd: str, timeout: int = 10) -> str:
    try:
        r = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            cwd=cwd, timeout=timeout,
        )
        out = r.stdout
        if r.stderr:
            out += f"\n[stderr] {r.stderr}"
        return out[:2000]  # cap tool result size
    except subprocess.TimeoutExpired:
        return "[error] command timed out"
    except Exception as e:
        return f"[error] {e}"


def build_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_agent(
    task: Task,
    model,
    tokenizer,
    sandbox_dir: str,
) -> TaskResult:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.instruction},
    ]

    total_tokens = 0
    total_time = 0.0
    total_generated_tokens = 0
    turns = 0
    error = None
    final_answer = ""

    for turn in range(MAX_TURNS):
        prompt = build_prompt(tokenizer, messages)
        prompt_tokens = count_tokens(tokenizer, prompt)
        remaining = MAX_CONTEXT_TOKENS - prompt_tokens - 20  # safety buffer

        if remaining < 50:
            error = "context exhausted"
            break

        max_gen = min(MAX_TOKENS_PER_STEP, remaining)

        t0 = time.time()
        output = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=max_gen,
            verbose=False,
        )
        elapsed = time.time() - t0

        gen_tokens = count_tokens(tokenizer, output)
        total_generated_tokens += gen_tokens
        total_time += elapsed
        turns += 1

        # Check for tool call
        tool_match = TOOL_CALL_RE.search(output)
        if tool_match:
            command = tool_match.group(1).strip()
            result = execute_bash(command, sandbox_dir)

            # Classify result: error vs success
            has_error = (
                result.startswith("[error]")
                or "[stderr]" in result
                or result.strip() == ""
            )

            # Append assistant turn + tool result
            assistant_content = output + f"\n```output\n{result}\n```"
            messages.append({"role": "assistant", "content": assistant_content})

            wrote_file = (
                "cat >" in command or ">" in command and "EOF" in command
                or ">>" in command or "echo" in command and ">" in command
            )

            if has_error:
                follow_up = (
                    "The command failed or returned no output. "
                    "Try a different approach — use a simpler command."
                )
            elif wrote_file:
                follow_up = (
                    "File written. Run it now to verify it works, then state the final answer in plain text."
                )
            else:
                follow_up = (
                    "Command executed. If you have enough information to answer, "
                    "state the final answer in plain text now (no code blocks). "
                    "Otherwise run another command."
                )
            messages.append({"role": "user", "content": follow_up})
        else:
            # No tool call — final answer
            final_answer = output
            messages.append({"role": "assistant", "content": output})
            if DEBUG:
                print(f"\n[debug turn={turn}] output:\n{output}\n")
            break

        if DEBUG:
            print(f"\n[debug turn={turn}] cmd: {command!r}\nresult: {result!r}\n")

    total_tokens = count_tokens(tokenizer, build_prompt(tokenizer, messages))
    tps = total_generated_tokens / total_time if total_time > 0 else 0.0

    try:
        success = task.validate(sandbox_dir, final_answer)
    except Exception as e:
        success = False
        error = f"validate error: {e}"

    return TaskResult(
        task=task.name,
        category=task.category,
        parallelizable=task.parallelizable,
        success=success,
        tokens_used=total_tokens,
        tps=round(tps, 1),
        turns=turns,
        context_headroom=MAX_CONTEXT_TOKENS - total_tokens,
        error=error,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

DEBUG = False


def run_tasks(tasks: list[Task], model, tokenizer) -> list[TaskResult]:
    results = []
    for task in tasks:
        print(f"  [{task.category}] {task.name} ... ", end="", flush=True)
        with tempfile.TemporaryDirectory() as sandbox:
            try:
                if task.setup:
                    task.setup(sandbox)
            except Exception as e:
                print(f"SETUP FAILED: {e}")
                results.append(TaskResult(
                    task=task.name, category=task.category,
                    parallelizable=task.parallelizable,
                    success=False, tokens_used=0, tps=0.0, turns=0,
                    context_headroom=MAX_CONTEXT_TOKENS, error=f"setup: {e}",
                ))
                continue

            result = run_agent(task, model, tokenizer, sandbox)
            results.append(result)

        status = "PASS" if result.success else "FAIL"
        print(f"{status} | {result.tokens_used} tok | {result.tps} tps | {result.turns} turns")

    return results


def measure_memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def print_summary(results: list[TaskResult]) -> None:
    passed = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    avg_tps = sum(r.tps for r in results if r.tps > 0) / max(1, len(results))
    avg_tokens = sum(r.tokens_used for r in results) / max(1, len(results))

    print("\n" + "=" * 60)
    print(f"RESULTS: {len(passed)}/{len(results)} passed")
    print(f"Avg TPS:    {avg_tps:.1f}")
    print(f"Avg tokens: {avg_tokens:.0f} / {MAX_CONTEXT_TOKENS}")
    print(f"Context OK: {sum(1 for r in results if r.context_headroom > 0)}/{len(results)}")

    if failed:
        print(f"\nFailed:")
        for r in failed:
            print(f"  {r.task}: {r.error or 'wrong answer'}")

    # Per-category breakdown
    categories = sorted(set(r.category for r in results))
    print("\nBy category:")
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        cat_passed = sum(1 for r in cat_results if r.success)
        print(f"  {cat:<15} {cat_passed}/{len(cat_results)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Merlin feasibility harness")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="MLX model ID or path")
    parser.add_argument("--task", help="Run a single task by name")
    parser.add_argument("--category", help="Run tasks matching category")
    parser.add_argument("--shard", help="Run shard i/N of tasks (e.g. 0/4, 1/4, ...)")
    parser.add_argument("--output", help="Save results JSON to file")
    parser.add_argument("--list", action="store_true", help="List available tasks")
    parser.add_argument("--debug", action="store_true", help="Print model output per task")
    args = parser.parse_args()

    global DEBUG
    if args.debug:
        DEBUG = True

    if args.list:
        for t in TASKS:
            parallel = "parallel" if t.parallelizable else "sequential"
            parent = f" [parent={t.tree_parent}]" if t.tree_parent else ""
            print(f"  {t.name:<35} {t.category:<15} {parallel}{parent}")
        return

    # Filter tasks
    tasks = TASKS
    if args.task:
        tasks = [t for t in tasks if t.name == args.task]
        if not tasks:
            print(f"Unknown task: {args.task}")
            sys.exit(1)
    elif args.category:
        tasks = [t for t in tasks if t.category == args.category]
        if not tasks:
            print(f"No tasks in category: {args.category}")
            sys.exit(1)

    if args.shard:
        try:
            i, n = map(int, args.shard.split("/"))
            assert 0 <= i < n
        except Exception:
            print("--shard must be i/N where 0 <= i < N (e.g. 0/4)")
            sys.exit(1)
        tasks = [t for idx, t in enumerate(tasks) if idx % n == i]
        if not tasks:
            print(f"Shard {i}/{n} is empty")
            sys.exit(1)

    print(f"Model:  {args.model}")
    print(f"Tasks:  {len(tasks)}")
    print(f"Ctx:    {MAX_CONTEXT_TOKENS} tokens\n")

    mem_before = measure_memory_mb()
    print(f"Loading model ... ", end="", flush=True)
    model, tokenizer = load(args.model)
    mem_after = measure_memory_mb()
    model_mem_mb = mem_after - mem_before
    print(f"done ({model_mem_mb:.0f} MB)\n")

    results = run_tasks(tasks, model, tokenizer)
    print_summary(results)

    if args.output:
        out_path = args.output
    else:
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, f"run_{int(time.time())}.json")

    with open(out_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "max_context_tokens": MAX_CONTEXT_TOKENS,
                "model_memory_mb": round(model_mem_mb, 1),
                "results": [asdict(r) for r in results],
            },
            f, indent=2,
        )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
