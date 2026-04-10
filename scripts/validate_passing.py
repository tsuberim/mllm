"""
Re-validate repos in passing.jsonl with the stricter baseline check
(must have at least 1 test that actually ran).

Rewrites passing.jsonl with only true positives.

Usage:
    python scripts/validate_passing.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.sandbox import Sandbox, SANDBOX_IMAGE
from harness.task_gen import _install_deps, _run_pytest_in_sandbox

PASSING = Path("data/repos/passing.jsonl")


def main():
    if not PASSING.exists():
        print("No passing.jsonl found")
        return

    repos = []
    with open(PASSING) as f:
        for line in f:
            repos.append(json.loads(line))

    print(f"Re-validating {len(repos)} repos...")
    confirmed = []

    for repo in repos:
        path = Path(repo["local_path"]).resolve()
        try:
            with Sandbox(repo_path=path, image=SANDBOX_IMAGE) as sb:
                _install_deps(sb, timeout=10)  # cached
                has_failures, _ = _run_pytest_in_sandbox(sb, timeout=60)
                status = "PASS" if not has_failures else "FALSE POSITIVE"
                print(f"  {status:15s} {repo['full_name']}")
                if not has_failures:
                    confirmed.append(repo)
        except Exception as e:
            print(f"  ERROR          {repo['full_name']}: {e}")

    with open(PASSING, "w") as f:
        for repo in confirmed:
            f.write(json.dumps(repo) + "\n")

    print(f"\n{len(confirmed)}/{len(repos)} confirmed passing → {PASSING}")


if __name__ == "__main__":
    main()
