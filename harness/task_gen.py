"""
Task generation from real repos using AST mutations.

Pipeline per repo:
  1. Start a sandbox with the repo mounted
  2. Install deps (pip install -e . or requirements.txt; npm ci for TS)
  3. Verify baseline: all tests pass
  4. For each candidate file: apply a random AST mutation, run tests
  5. If ≥1 test fails → valid task; save mutated repo copy for the agent trace

The sandbox approach is required because repos need their deps installed before
tests can collect and run.
"""

import random
import re
import shutil
import subprocess
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
    repo_commit: str = ""         # git SHA of the cloned repo (for RL replay)
    mutation_patch: str = ""      # unified diff: original → mutated (for RL replay)
    # Snapshot of test file contents at task creation time.
    # Restored before validation to prevent the agent from gaming the reward
    # by modifying tests. Keys are relative paths (str), values are file contents.
    test_snapshots: dict = field(default_factory=dict)

    def setup(self, sandbox_dir: str) -> None:
        """Copy the mutated repo into sandbox_dir."""
        dest = Path(sandbox_dir)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self.repo_path, dest, symlinks=True)

    def validate(self, sandbox, answer: str | None) -> bool:
        """Run tests in the sandbox. Pass iff all tests pass and no test files were touched.

        Penalizes test tampering: if the agent modified any test file the task
        fails immediately, regardless of whether tests would pass.
        Restores test files before running tests so the reward is always based
        on the original test suite.
        """
        root = sandbox.repo_path
        # Check for tampering: modified or new test files both count as cheating
        known = set(self.test_snapshots)
        test_pats_py = ("test_*.py", "*_test.py", "conftest.py")
        test_pats_ts = ("*.test.ts", "*.spec.ts", "*.test.tsx", "*.spec.tsx")
        for pat in test_pats_py + test_pats_ts:
            for tf in root.rglob(pat):
                rel = str(tf.relative_to(root))
                if rel not in known:
                    return False  # penalty: agent created a new test file
                try:
                    if tf.read_text(errors="replace") != self.test_snapshots[rel]:
                        return False  # penalty: agent modified an existing test file
                except Exception:
                    pass
        # Restore snapshot so tests run against original suite
        for rel, content in self.test_snapshots.items():
            target = root / rel
            try:
                target.write_text(content)
            except Exception:
                pass

        # Detect language and run appropriate test runner
        if (root / "package.json").exists():
            result = sandbox.bash(
                "npx jest --no-coverage --forceExit --testTimeout=5000 2>&1 | tail -3",
                timeout=90,
            )
        else:
            result = sandbox.bash(
                ".venv/bin/python -m pytest --tb=no -q --continue-on-collection-errors 2>&1 | tail -1",
                timeout=60,
            )
        return result.exit_code == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_ts_repo(repo_path: Path) -> bool:
    """True if the repo is TypeScript-based (has package.json + .ts files)."""
    return (repo_path / "package.json").exists() and any(repo_path.rglob("*.ts"))


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


def _find_ts_files(repo_path: Path, min_lines: int = 20) -> list[Path]:
    """Find non-test TypeScript source files with enough code to mutate."""
    _SKIP_DIRS = {".git", "node_modules", "dist", "build", ".next", "coverage", "__tests__"}
    candidates = []
    for f in repo_path.rglob("*.ts"):
        parts = set(f.parts)
        if parts & _SKIP_DIRS:
            continue
        name = f.name
        if ".test." in name or ".spec." in name or name.endswith(".d.ts"):
            continue
        try:
            lines = f.read_text(errors="replace").count("\n")
            if lines >= min_lines:
                candidates.append(f)
        except Exception:
            continue
    return candidates


def _install_ts_deps(sandbox: Sandbox, timeout: int = 120) -> bool:
    """Install TypeScript/Node deps via npm ci (or npm install as fallback)."""
    r = sandbox.bash("[ -d node_modules ] && echo cached || echo fresh", timeout=10)
    if r.stdout.strip() == "cached":
        return True
    r = sandbox.bash("npm ci --prefer-offline --ignore-scripts 2>&1 | tail -3", timeout=timeout)
    if r.exit_code != 0:
        sandbox.bash("npm install --ignore-scripts 2>&1 | tail -3", timeout=timeout)
    return True


def _run_jest_in_sandbox(sandbox: Sandbox, timeout: int = 90, runner: str = "jest") -> tuple[bool, list[str]]:
    """Run jest/vitest. Returns (any_failures, failing_test_ids)."""
    if runner == "vitest":
        cmd = "npx vitest run --reporter=verbose 2>&1"
    else:
        cmd = "npx jest --no-coverage --forceExit --testTimeout=5000 2>&1"

    r = sandbox.bash(cmd, timeout=timeout)
    failing = []
    n_passed = 0
    has_failures = False
    for line in r.stdout.splitlines():
        line = line.strip()
        m = re.search(r"(\d+) passed", line)
        if m:
            n_passed = max(n_passed, int(m.group(1)))
        if re.search(r"\d+ failed", line):
            has_failures = True
        # jest verbose failure lines: "● <test name>"
        if line.startswith("● ") and not line.startswith("● Test suite"):
            failing.append(line[2:].strip())

    if n_passed == 0 and not has_failures:
        return True, []  # no tests ran
    return has_failures, failing


def _detect_ts_runner(repo_path: Path) -> str:
    """Return 'jest' or 'vitest' based on package.json devDependencies."""
    import json
    try:
        pkg = json.loads((repo_path / "package.json").read_text(errors="replace"))
        deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
        if "vitest" in deps:
            return "vitest"
    except Exception:
        pass
    return "jest"


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
    sandbox_class=None,
) -> list[GeneratedTask]:
    """
    Generate up to n_tasks fix-the-bug tasks from a repo.
    Supports Python (pytest) and TypeScript (jest/vitest) repos.

    sandbox_class: Sandbox (Docker, default) or LocalSandbox (subprocess, for Modal).
    """
    import difflib
    if sandbox_class is None:
        sandbox_class = Sandbox

    repo_path = Path(repo_path).resolve()
    rng = random.Random(seed)

    if _is_ts_repo(repo_path):
        return _generate_ts_tasks(
            repo_path, n_tasks, max_attempts, rng, sandbox_image, sandbox_class,
        )
    return _generate_py_tasks(
        repo_path, n_tasks, max_attempts, rng, sandbox_image, sandbox_class,
    )


def _capture_test_snapshots(repo_path: Path, is_ts: bool = False) -> dict[str, str]:
    snapshots: dict[str, str] = {}
    pats = (
        ["*.test.ts", "*.spec.ts", "*.test.tsx", "*.spec.tsx"]
        if is_ts else
        ["test_*.py", "*_test.py", "conftest.py"]
    )
    for pat in pats:
        for tf in repo_path.rglob(pat):
            rel = str(tf.relative_to(repo_path))
            try:
                snapshots[rel] = tf.read_text(errors="replace")
            except Exception:
                pass
    return snapshots


def _make_task(repo_path: Path, rel_path: Path, mutation, failing_tests: list[str],
               repo_commit: str) -> GeneratedTask:
    import difflib
    is_ts = rel_path.suffix in (".ts", ".tsx")
    snapshot_dir = Path(tempfile.mkdtemp()) / "repo"
    shutil.copytree(repo_path, snapshot_dir, symlinks=True)
    (snapshot_dir / rel_path).write_text(mutation.mutated)

    test_snapshots = _capture_test_snapshots(repo_path, is_ts=is_ts)
    mutation_patch = "".join(difflib.unified_diff(
        mutation.original.splitlines(keepends=True),
        mutation.mutated.splitlines(keepends=True),
        fromfile=f"a/{rel_path}",
        tofile=f"b/{rel_path}",
    ))
    return GeneratedTask(
        name=f"{repo_path.name}__{rel_path.stem}__{mutation.kind}",
        repo_path=str(snapshot_dir),
        instruction=_format_instruction(str(rel_path), failing_tests),
        mutation=mutation,
        mutated_file=str(rel_path),
        failing_tests=failing_tests,
        repo_commit=repo_commit,
        mutation_patch=mutation_patch,
        test_snapshots=test_snapshots,
    )


def _get_repo_commit(repo_path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_path), text=True,
        ).strip()
    except Exception:
        return ""


def _generate_py_tasks(
    repo_path: Path, n_tasks: int, max_attempts: int,
    rng: random.Random, sandbox_image: str, sandbox_class,
) -> list[GeneratedTask]:
    repo_commit = _get_repo_commit(repo_path)
    py_files = _find_py_files(repo_path)
    if not py_files:
        return []

    tasks = []
    try:
        with sandbox_class(repo_path=repo_path, image=sandbox_image) as sb:
            _install_deps(sb)
            has_failures, _ = _run_pytest_in_sandbox(sb)
            if has_failures:
                return []

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

                original = py_file.read_bytes()
                py_file.write_text(mutation.mutated)
                try:
                    has_failures, failing_tests = _run_pytest_in_sandbox(sb)
                finally:
                    py_file.write_bytes(original)

                if not has_failures:
                    attempts += 1
                    continue

                tasks.append(_make_task(repo_path, rel_path, mutation, failing_tests, repo_commit))
                attempts += 1
    except Exception:
        pass
    return tasks


def _generate_ts_tasks(
    repo_path: Path, n_tasks: int, max_attempts: int,
    rng: random.Random, sandbox_image: str, sandbox_class,
) -> list[GeneratedTask]:
    from .ts_mutator import mutate_ts, AVAILABLE as TS_AVAILABLE
    if not TS_AVAILABLE:
        return []

    repo_commit = _get_repo_commit(repo_path)
    ts_runner = _detect_ts_runner(repo_path)
    ts_files = _find_ts_files(repo_path)
    if not ts_files:
        return []

    tasks = []
    try:
        with sandbox_class(repo_path=repo_path, image=sandbox_image) as sb:
            _install_ts_deps(sb)
            has_failures, _ = _run_jest_in_sandbox(sb, runner=ts_runner)
            if has_failures:
                return []

            rng.shuffle(ts_files)
            attempts = 0
            for ts_file in ts_files:
                if len(tasks) >= n_tasks or attempts >= max_attempts:
                    break
                rel_path = ts_file.relative_to(repo_path)
                source = ts_file.read_text(errors="replace")
                mutation = mutate_ts(source, seed=rng.randint(0, 2**32))
                if mutation is None:
                    attempts += 1
                    continue

                original = ts_file.read_bytes()
                ts_file.write_text(mutation.mutated)
                try:
                    has_failures, failing_tests = _run_jest_in_sandbox(sb, runner=ts_runner)
                finally:
                    ts_file.write_bytes(original)

                if not has_failures:
                    attempts += 1
                    continue

                tasks.append(_make_task(repo_path, rel_path, mutation, failing_tests, repo_commit))
                attempts += 1
    except Exception:
        pass
    return tasks
