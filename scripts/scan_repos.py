"""
Scan all repos in data/repos/meta.jsonl and find ones with passing test suites.

Runs pytest inside a merlin-sandbox container. Installs deps into .venv/ inside
the repo dir (persists on host — skipped on subsequent runs).

Writes passing repos to data/repos/passing.jsonl.

Usage:
    python scripts/scan_repos.py [--workers N]
"""

import argparse
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.sandbox import Sandbox, SANDBOX_IMAGE
from harness.task_gen import _install_deps, _run_pytest_in_sandbox

META    = Path("data/repos/meta.jsonl")
OUT     = Path("data/repos/passing.jsonl")


def check_repo(repo: dict) -> tuple[bool, str]:
    """Returns (passes, status_message)."""
    path = Path(repo["local_path"]).resolve()
    try:
        with Sandbox(repo_path=path, image=SANDBOX_IMAGE) as sb:
            _install_deps(sb, timeout=120)
            has_failures, failing = _run_pytest_in_sandbox(sb, timeout=60)
            if not has_failures:
                return True, "PASS"
            return False, f"fail ({len(failing)} failures)"
    except Exception as e:
        return False, f"err: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4, help="Parallel containers")
    args = parser.parse_args()

    repos = []
    with open(META) as f:
        for line in f:
            repos.append(json.loads(line))

    # Skip already-known passing repos
    known_passing = set()
    if OUT.exists():
        with open(OUT) as f:
            for line in f:
                known_passing.add(json.loads(line)["full_name"])

    to_scan = [r for r in repos if r["full_name"] not in known_passing]
    print(f"Repos: {len(repos)} total, {len(known_passing)} already passing, {len(to_scan)} to scan")
    print(f"Workers: {args.workers}\n")

    out_lock = threading.Lock()
    passing_count = [len(known_passing)]
    idx_lock = threading.Lock()
    idx = [0]

    def worker():
        while True:
            with idx_lock:
                i = idx[0]
                if i >= len(to_scan):
                    return
                idx[0] += 1
            repo = to_scan[i]
            passes, status = check_repo(repo)
            n = i + 1
            print(f"[{n:3d}/{len(to_scan)}] {status:30s} {repo['full_name']}", flush=True)
            if passes:
                with out_lock:
                    with open(OUT, "a") as f:
                        f.write(json.dumps(repo) + "\n")
                    passing_count[0] += 1

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = len(repos)
    print(f"\nDone. {passing_count[0]}/{total} repos pass → {OUT}")


if __name__ == "__main__":
    main()
