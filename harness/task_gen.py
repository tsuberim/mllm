"""
Task generation from real repos using AST mutations.

Pipeline per repo:
  1. Start a sandbox with the repo mounted
  2. Install deps (pip install -e . or requirements.txt)
  3. Verify baseline: all tests pass
  4. For each candidate file: apply a random AST mutation, run pytest
  5. If ≥1 test fails → valid task; save mutated repo copy for the agent trace

The sandbox approach is required because repos need their deps installed before
pytest can collect and run tests.
"""

import random
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .mutator import mutate, Mutation
from .sandbox import Sandbox, SANDBOX_IMAGE


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@dataclass
class GeneratedTask:
    name: str
    category: str = "fix_tests"
    parallelizable: bool = True
    repo_path: str = ""           # path to the mutated repo snapshot
    instruction: str = ""
    mutation: Mutation = None
    mutated_file: str = ""        # relative path of mutated file
    failing_tests: list[str] = field(default_factory=list)

    def setup(self, sandbox_dir: str) -> None:
        """Copy the mutated repo into sandbox_dir."""
        dest = Path(sandbox_dir)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self.repo_path, dest, symlinks=True)

    def validate(self, sandbox, answer: str | None) -> bool:
        """Run pytest in the sandbox. Pass iff all tests pass."""
        result = sandbox.bash(
            ".venv/bin/python -m pytest --tb=no -q --continue-on-collection-errors 2>&1 | tail -1",
            timeout=60,
        )
        return result.exit_code == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_py_files(repo_path: Path, min_functions: int = 2) -> list[Path]:
    """Find non-test Python files that have enough functions to mutate."""
    import ast
    candidates = []
    for f in repo_path.rglob("*.py"):
        parts = f.parts
        if any(p in ("test", "tests", "migrations", "__pycache__", ".git", ".venv") for p in parts):
            continue
        if f.name.startswith("test_") or f.name.endswith("_test.py"):
            continue
        try:
            source = f.read_text(errors="replace")
            tree = ast.parse(source)
            n_funcs = sum(
                1 for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            if n_funcs >= min_functions:
                candidates.append(f)
        except Exception:
            continue
    return candidates


def _install_deps(sandbox: Sandbox, timeout: int = 120) -> bool:
    """Install repo dependencies into .venv/ inside the repo dir (persists on host volume).

    Skips install if .venv/ already exists from a previous run.
    Installs runtime deps + common test requirement files.
    """
    # .venv.testdeps marks that test deps were installed; plain .venv/bin/python is not enough
    r = sandbox.bash("[ -f .venv/.testdeps ] && echo cached || echo fresh", timeout=10)
    if r.stdout.strip() == "cached":
        return True

    # Create venv and install pytest
    sandbox.bash(
        "python -m venv .venv && .venv/bin/pip install pytest -q --no-warn-script-location 2>&1 | tail -1",
        timeout=timeout,
    )
    # Install package itself (runtime deps)
    sandbox.bash(
        ".venv/bin/pip install -e . -q --no-warn-script-location 2>&1 | tail -3",
        timeout=timeout,
    )
    # Install all *requirements*.txt files found anywhere in the repo
    sandbox.bash(
        "find . -name '*requirements*.txt' -not -path './.venv/*' "
        "| xargs -I{} .venv/bin/pip install -r {} -q --no-warn-script-location 2>&1 | tail -3 || true",
        timeout=timeout,
    )
    sandbox.bash("touch .venv/.testdeps", timeout=5)
    return True  # best-effort; pytest will tell us if it fails


def _run_pytest_in_sandbox(sandbox: Sandbox, timeout: int = 60, python: str = ".venv/bin/python") -> tuple[bool, list[str]]:
    """Run pytest inside sandbox. Returns (any_failures, failing_test_ids).

    Exit codes:
      0 = all passed
      1 = some tests failed  ← the only case we treat as "has_failures"
      2 = interrupted (collection error, bad args)  ← skip repo
      3 = internal error
      4 = usage error
      5 = no tests collected
    We use --continue-on-collection-errors so partial import failures don't
    abort the whole run.
    """
    r = sandbox.bash(
        f"{python} -m pytest --tb=no -q --no-header --continue-on-collection-errors 2>&1",
        timeout=timeout,
    )
    failing = []
    n_passed = 0
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("FAILED "):
            failing.append(line[len("FAILED "):].split(" - ")[0].strip())
        # Match lines like "5 passed" or "3 passed, 2 warnings"
        m = re.search(r"(\d+) passed", line)
        if m:
            n_passed = int(m.group(1))

    # exit_code 5 = no tests collected; 0 with zero passed = all collection errors
    # exit_code 1 can mean collection errors only (no FAILED lines) — don't trust it alone
    if r.exit_code == 5 or (n_passed == 0 and not failing):
        return True, []  # no tests ran → skip repo

    # Only count explicit FAILED lines as test failures; ignore collection errors (exit 1, no FAILED)
    has_failures = bool(failing)
    return has_failures, failing


def _format_instruction(mutated_file: str, failing_tests: list[str]) -> str:
    lines = [f"The test suite is failing. There is a bug in `{mutated_file}`."]
    if failing_tests:
        lines.append(f"Failing tests ({len(failing_tests)}):")
        for t in failing_tests[:5]:
            lines.append(f"  - {t}")
        if len(failing_tests) > 5:
            lines.append(f"  ... and {len(failing_tests) - 5} more")
    lines.append("Fix the bug so all tests pass.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

def generate_tasks(
    repo_path: str | Path,
    n_tasks: int = 5,
    max_attempts: int = 50,
    seed: int | None = None,
    sandbox_image: str = SANDBOX_IMAGE,
) -> list[GeneratedTask]:
    """
    Generate up to n_tasks fix-the-bug tasks from a repo.
    Runs pytest inside a Docker sandbox so deps are available.
    """
    repo_path = Path(repo_path).resolve()
    rng = random.Random(seed)

    py_files = _find_py_files(repo_path)
    if not py_files:
        return []

    tasks = []

    # One sandbox for the whole repo — install deps once, reuse
    try:
        with Sandbox(repo_path=repo_path, image=sandbox_image) as sb:
            _install_deps(sb)

            # Verify baseline: repo must have passing tests
            has_failures, _ = _run_pytest_in_sandbox(sb)
            if has_failures:
                return []  # already broken or no tests

            # Try mutations
            rng.shuffle(py_files)
            attempts = 0

            for py_file in py_files:
                if len(tasks) >= n_tasks or attempts >= max_attempts:
                    break

                rel_path = py_file.relative_to(repo_path)
                source = py_file.read_text(errors="replace")
                mutation = mutate(source, seed=rng.randint(0, 2**32))
                if mutation is None:
                    attempts += 1
                    continue

                # Write mutation directly into the mounted repo
                original = py_file.read_bytes()
                py_file.write_text(mutation.mutated)

                try:
                    has_failures, failing_tests = _run_pytest_in_sandbox(sb)
                finally:
                    # Always revert, even on exception
                    py_file.write_bytes(original)

                if not has_failures:
                    attempts += 1
                    continue

                # Valid mutation — save a snapshot of the mutated repo
                snapshot_dir = Path(tempfile.mkdtemp()) / "repo"
                shutil.copytree(repo_path, snapshot_dir, symlinks=True)
                # Apply mutation to snapshot (clean repo was restored above)
                (snapshot_dir / rel_path).write_text(mutation.mutated)

                task = GeneratedTask(
                    name=f"{repo_path.name}__{rel_path.stem}__{mutation.kind}",
                    repo_path=str(snapshot_dir),
                    instruction=_format_instruction(str(rel_path), failing_tests),
                    mutation=mutation,
                    mutated_file=str(rel_path),
                    failing_tests=failing_tests,
                )
                tasks.append(task)
                attempts += 1

    except Exception:
        pass  # sandbox failed to start or crashed

    return tasks
