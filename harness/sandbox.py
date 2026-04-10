"""
Docker sandbox management.

Each trace runs in an isolated container — stateful within a trace, fresh per trace.
The sandbox image is pre-built with all required tools and Python packages.

Usage:
    with Sandbox(repo_path="/data/repos/owner__repo", image="merlin-sandbox") as sb:
        result = sb.bash("grep -rn 'TODO' src/")
        print(result["stdout"], result["exit_code"])
"""

import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# Truncation limits — keep bash output within model context budget
MAX_STDOUT_LINES = 200
MAX_STDERR_LINES = 50
TRUNCATION_MARKER = "# ... {n} more lines"

SANDBOX_IMAGE = "merlin-sandbox"
BASH_TIMEOUT_S  = 30    # per-call timeout
TRACE_TIMEOUT_S = 300   # per-trace timeout (5 min)


@dataclass
class BashResult:
    stdout: str
    stderr: str
    exit_code: int
    truncated: bool = False


def _truncate(text: str, max_lines: int) -> tuple[str, bool]:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text, False
    kept = lines[:max_lines]
    remaining = len(lines) - max_lines
    kept.append(TRUNCATION_MARKER.format(n=remaining))
    return "\n".join(kept), True


class Sandbox:
    def __init__(
        self,
        repo_path: str | Path,
        image: str = SANDBOX_IMAGE,
        network: bool = True,   # allow network; needed for curl in traces
    ):
        self.repo_path = Path(repo_path).resolve()
        self.image = image
        self.network = network
        self._container_id: str | None = None
        self._lock = threading.Lock()

    def start(self):
        """Start the sandbox container."""
        network_flag = "bridge" if self.network else "none"
        cmd = [
            "docker", "run", "-d",
            "--rm",
            "-v", f"{self.repo_path}:/workspace",
            "-w", "/workspace",
            "-e", "VIRTUAL_ENV=/workspace/.venv",
            "-e", "PATH=/workspace/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            f"--network={network_flag}",
            "--memory=2g",
            "--cpus=2",
            self.image,
            "sleep", "infinity",
        ]
        self._container_id = subprocess.check_output(cmd, text=True).strip()

    def stop(self):
        """Stop and remove the container."""
        if self._container_id:
            subprocess.run(
                ["docker", "stop", "-t", "5", self._container_id],
                capture_output=True,
            )
            self._container_id = None

    def bash(self, cmd: str, timeout: int = BASH_TIMEOUT_S) -> BashResult:
        """Execute a bash command inside the sandbox."""
        if not self._container_id:
            raise RuntimeError("Sandbox not started")

        with self._lock:
            try:
                result = subprocess.run(
                    ["docker", "exec", self._container_id, "bash", "-c", cmd],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                stdout, stdout_trunc = _truncate(result.stdout, MAX_STDOUT_LINES)
                stderr, stderr_trunc = _truncate(result.stderr, MAX_STDERR_LINES)
                return BashResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=result.returncode,
                    truncated=stdout_trunc or stderr_trunc,
                )
            except subprocess.TimeoutExpired:
                return BashResult(
                    stdout="",
                    stderr=f"# command timed out after {timeout}s",
                    exit_code=124,
                    truncated=False,
                )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
