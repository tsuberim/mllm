"""
Async trace generation runner.

Pulls tasks from a queue, runs agent loops in parallel using a thread pool,
writes successful traces to output JSONL.

Usage:
    # Generate tasks from real repos (AST mutations):
    python -m harness.runner \
        --repos data/repos/ \
        --out data/traces/pilot.jsonl \
        --workers 50 \
        --tasks-per-repo 5 \
        --vllm-url http://localhost:8000/v1

    # Use a static tasks.py file:
    python -m harness.runner \
        --tasks-file research/feasibility/tasks.py \
        --out data/traces/pilot.jsonl \
        --workers 50 \
        --vllm-url http://localhost:8000/v1
"""

import argparse
import importlib.util
import json
import queue
import sys
import threading
import time
from pathlib import Path

from .agent import AgentConfig, run_agent
from .protocol import strip_thinking
from .sandbox import Sandbox, SANDBOX_IMAGE
from .task_gen import generate_tasks


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def load_tasks_file(tasks_path: str) -> list:
    """Load tasks from a tasks.py file (expects TASKS list of Task objects)."""
    spec = importlib.util.spec_from_file_location("tasks", tasks_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TASKS


def load_repos(path: str) -> list[dict]:
    """Load repo metadata from a .jsonl file or a directory containing meta.jsonl."""
    p = Path(path)
    jsonl = p if p.suffix == ".jsonl" else p / "meta.jsonl"
    if not jsonl.exists():
        return []
    repos = []
    with open(jsonl) as f:
        for line in f:
            repos.append(json.loads(line))
    return repos


def build_task_queue_from_repos(
    repos: list[dict],
    tasks_per_repo: int,
    seed: int = 42,
    gen_workers: int = 4,
) -> queue.Queue:
    """Generate AST-mutation tasks from repos and return a populated queue.

    Uses a thread pool to generate tasks from multiple repos in parallel.
    gen_workers controls how many Docker sandbox containers run simultaneously.
    """
    import random
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rng = random.Random(seed)
    task_q: queue.Queue = queue.Queue()
    print(f"Generating tasks from {len(repos)} repos ({tasks_per_repo} tasks/repo, {gen_workers} workers)...")

    def _gen_for_repo(repo_meta, repo_seed):
        repo_path = repo_meta.get("local_path", "")
        if not repo_path or not Path(repo_path).exists():
            return [], repo_meta
        tasks = generate_tasks(repo_path=repo_path, n_tasks=tasks_per_repo, seed=repo_seed)
        return tasks, repo_meta

    seeds = [rng.randint(0, 2**32) for _ in repos]
    with ThreadPoolExecutor(max_workers=gen_workers) as ex:
        futures = {ex.submit(_gen_for_repo, repo, s): repo for repo, s in zip(repos, seeds)}
        for fut in as_completed(futures):
            tasks, repo_meta = fut.result()
            for task in tasks:
                task_q.put((task, repo_meta))
            if tasks:
                print(f"  {repo_meta['full_name']}: {len(tasks)} tasks")

    print(f"Generated {task_q.qsize()} tasks total")
    return task_q


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(
    task_q: queue.Queue,
    out_path: Path,
    out_lock: threading.Lock,
    config: AgentConfig,
    repos: list[dict],
    stats: dict,
    stats_lock: threading.Lock,
    sandbox_image: str,
):
    import random

    while True:
        try:
            item = task_q.get_nowait()
        except queue.Empty:
            break

        task, repo_meta = item

        # GeneratedTask has its own snapshot dir with the mutation already applied.
        # Use it directly as the sandbox root — no copying needed, avoids races
        # between parallel workers sharing the same original repo path.
        sandbox_dir = getattr(task, "repo_path", None) or (
            repo_meta["local_path"] if repo_meta else "/tmp/empty_sandbox"
        )

        try:
            with Sandbox(repo_path=sandbox_dir, image=sandbox_image) as sb:
                validator = task.validate if hasattr(task, "validate") else None

                result = run_agent(
                    task=task.instruction,
                    sandbox=sb,
                    config=config,
                    repo_name=repo_meta["full_name"] if repo_meta else "",
                    validator=validator,
                )
        except Exception as e:
            result = None
            with stats_lock:
                stats["errors"] += 1
            print(f"[worker] exception on task {task.name}: {e}")
            task_q.task_done()
            continue

        with stats_lock:
            stats["total"] += 1
            if result.success:
                stats["success"] += 1
            else:
                stats["failed"] += 1
                stats["failure_reasons"][result.failure_reason] = (
                    stats["failure_reasons"].get(result.failure_reason, 0) + 1
                )

        if result.success:
            record = {
                "task": result.task,
                "task_name": task.name,
                "task_category": task.category,
                "repo": result.repo,
                "trace": result.trace,
                "trace_sft": strip_thinking(result.trace),
                "answer": result.answer,
                "n_tool_calls": result.n_tool_calls,
                "n_tokens": result.n_tokens,
                "duration_s": round(result.duration_s, 2),
            }
            with out_lock:
                with open(out_path, "a") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with stats_lock:
            total = stats["total"]
            success = stats["success"]
            errors = stats["errors"]
            n_q = task_q.qsize()
            icon = "✓" if result.success else "✗"
            print(
                f"[{total:3d}] {icon} {task.name[:50]:50s} | "
                f"ok={success} err={errors} q={n_q} | "
                f"{result.duration_s:.0f}s",
                flush=True,
            )

        task_q.task_done()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(
    out_path: Path,
    n_workers: int,
    config: AgentConfig,
    sandbox_image: str = SANDBOX_IMAGE,
    # repo-based generation
    repos_dir: str | None = None,
    tasks_per_repo: int = 5,
    gen_workers: int = 4,
    # static tasks file
    tasks_file: str | None = None,
    repeats: int = 1,
):
    repos = []
    if repos_dir:
        # Prefer passing.jsonl (pre-scanned) over full meta.jsonl
        passing_path = Path(repos_dir) / "passing.jsonl"
        repos = load_repos(str(passing_path) if passing_path.exists() else repos_dir)
        task_q = build_task_queue_from_repos(repos, tasks_per_repo, gen_workers=gen_workers)
    elif tasks_file:
        tasks = load_tasks_file(tasks_file)
        task_q: queue.Queue = queue.Queue()
        for _ in range(repeats):
            for task in tasks:
                task_q.put((task, None))
        print(f"Tasks: {task_q.qsize()} | Workers: {n_workers}")
    else:
        raise ValueError("Either --repos or --tasks-file required")

    print(f"Total task instances: {task_q.qsize()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_lock = threading.Lock()
    stats = {"total": 0, "success": 0, "failed": 0, "errors": 0, "failure_reasons": {}}
    stats_lock = threading.Lock()

    threads = []
    for _ in range(n_workers):
        t = threading.Thread(
            target=_worker,
            args=(task_q, out_path, out_lock, config, repos, stats, stats_lock, sandbox_image),
            daemon=True,
        )
        t.start()
        threads.append(t)

    task_q.join()
    for t in threads:
        t.join(timeout=1)

    total = stats["total"]
    success = stats["success"]
    print(f"\n--- Results ---")
    print(f"Total:   {total}")
    print(f"Success: {success} ({100*success/max(total,1):.1f}%)")
    print(f"Failed:  {stats['failed']}")
    print(f"Errors:  {stats['errors']}")
    if stats["failure_reasons"]:
        print("Failure reasons:")
        for reason, count in sorted(stats["failure_reasons"].items(), key=lambda x: -x[1]):
            print(f"  {count:4d}  {reason}")
    print(f"\nOutput: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate agentic traces")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-trace-tokens", type=int, default=4096)
    parser.add_argument("--sandbox-image", default=SANDBOX_IMAGE)
    # repo-based generation
    parser.add_argument("--repos", help="Repos directory with meta.jsonl (AST mutation tasks)")
    parser.add_argument("--tasks-per-repo", type=int, default=5)
    parser.add_argument("--gen-workers", type=int, default=4, help="Parallel workers for task generation")
    # static tasks file
    parser.add_argument("--tasks-file", help="Path to tasks.py with TASKS list")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    if not args.repos and not args.tasks_file:
        parser.error("Either --repos or --tasks-file required")

    config = AgentConfig(
        vllm_url=args.vllm_url,
        model=args.model,
        temperature=args.temperature,
        max_trace_tokens=args.max_trace_tokens,
    )

    run(
        out_path=Path(args.out),
        n_workers=args.workers,
        config=config,
        sandbox_image=args.sandbox_image,
        repos_dir=args.repos,
        tasks_per_repo=args.tasks_per_repo,
        gen_workers=args.gen_workers,
        tasks_file=args.tasks_file,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
