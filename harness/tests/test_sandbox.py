"""Tests for Docker sandbox — requires Docker and merlin-sandbox image."""

import tempfile
from pathlib import Path

import pytest
from harness.sandbox import Sandbox, MAX_STDOUT_LINES, TRUNCATION_MARKER, _truncate


# ---------------------------------------------------------------------------
# Unit tests (no Docker)
# ---------------------------------------------------------------------------

def test_truncate_short():
    text = "line1\nline2\nline3"
    out, trunc = _truncate(text, max_lines=10)
    assert out == text
    assert not trunc


def test_truncate_long():
    lines = [f"line{i}" for i in range(20)]
    text = "\n".join(lines)
    out, trunc = _truncate(text, max_lines=5)
    assert trunc
    assert "line0" in out
    assert "line4" in out
    assert "line5" not in out
    assert "15 more lines" in out


def test_truncate_exact_limit():
    lines = [f"line{i}" for i in range(10)]
    text = "\n".join(lines)
    out, trunc = _truncate(text, max_lines=10)
    assert not trunc
    assert out == text


# ---------------------------------------------------------------------------
# Integration tests (require Docker + merlin-sandbox image)
# ---------------------------------------------------------------------------

@pytest.fixture
def repo(tmp_path):
    """A minimal repo directory for sandbox testing."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def foo():\n    pass\n")
    (tmp_path / "src" / "utils.py").write_text("# TODO: fix this\n")
    return tmp_path


@pytest.mark.docker
def test_sandbox_basic_bash(repo):
    with Sandbox(repo_path=repo) as sb:
        result = sb.bash("echo hello")
    assert result.stdout.strip() == "hello"
    assert result.exit_code == 0


@pytest.mark.docker
def test_sandbox_grep(repo):
    with Sandbox(repo_path=repo) as sb:
        result = sb.bash("grep -rn 'TODO' src/")
    assert "TODO" in result.stdout
    assert result.exit_code == 0


@pytest.mark.docker
def test_sandbox_nonzero_exit(repo):
    with Sandbox(repo_path=repo) as sb:
        result = sb.bash("ls /nonexistent_path_xyz")
    assert result.exit_code != 0
    assert result.stderr or result.stdout  # some error output


@pytest.mark.docker
def test_sandbox_stateful(repo):
    """State persists across bash calls within the same sandbox."""
    with Sandbox(repo_path=repo) as sb:
        sb.bash("echo 'hello' > /tmp/testfile.txt")
        result = sb.bash("cat /tmp/testfile.txt")
    assert "hello" in result.stdout


@pytest.mark.docker
def test_sandbox_timeout(repo):
    with Sandbox(repo_path=repo) as sb:
        result = sb.bash("sleep 60", timeout=1)
    assert result.exit_code == 124
    assert "timed out" in result.stderr


@pytest.mark.docker
def test_sandbox_python(repo):
    with Sandbox(repo_path=repo) as sb:
        result = sb.bash("python3 -c \"print('hello from python')\"")
    assert "hello from python" in result.stdout
    assert result.exit_code == 0


@pytest.mark.docker
def test_sandbox_truncation(repo):
    with Sandbox(repo_path=repo) as sb:
        # Generate more than MAX_STDOUT_LINES lines
        result = sb.bash(f"seq 1 {MAX_STDOUT_LINES + 50}")
    assert result.truncated
    assert TRUNCATION_MARKER.split("{")[0] in result.stdout


@pytest.mark.docker
def test_sandbox_tools_available(repo):
    """Verify key tools are installed in the sandbox image."""
    tools = ["git", "curl", "rg", "fd", "jq", "yq", "python3", "sqlite3"]
    with Sandbox(repo_path=repo) as sb:
        for tool in tools:
            result = sb.bash(f"which {tool}")
            assert result.exit_code == 0, f"{tool} not found in sandbox"
