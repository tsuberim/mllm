"""
Task definitions for Merlin feasibility validation.

Each task is self-contained: setup creates an isolated sandbox, validate checks
the result without external state. Tasks are grouped into categories reflecting
real agentic coding workloads. Tree tasks decompose into self-contained leaves
that can be dispatched in parallel by an orchestrator.
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Task:
    name: str
    category: str          # search | analysis | transform | scripting | multi_step | tree_leaf | tree_root | semantic_search
    parallelizable: bool
    instruction: str       # clean orchestrator-style instruction
    setup: Callable[[str], None]
    validate: Callable[[str, str], bool]  # (sandbox_dir, model_answer) -> correct
    tree_parent: Optional[str] = None     # name of parent task if this is a tree leaf
    expected_tokens: Optional[int] = None # rough expectation for context usage


def _w(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _run(cmd: str, cwd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    return r.stdout.strip()


# ---------------------------------------------------------------------------
# SEARCH
# ---------------------------------------------------------------------------

TASKS = []

TASKS += [
    Task(
        name="find_py_files",
        category="search",
        parallelizable=True,
        instruction="List all .py files in the current directory tree. Output one path per line, no extra text.",
        setup=lambda d: [
            _w(f"{d}/main.py", "print('hello')\n"),
            _w(f"{d}/utils/helpers.py", "def foo(): pass\n"),
            _w(f"{d}/utils/math.py", "def add(a, b): return a+b\n"),
            _w(f"{d}/README.md", "# Project\n"),
            _w(f"{d}/config.json", "{}\n"),
        ],
        validate=lambda d, ans: (
            "main.py" in ans and "helpers.py" in ans and "math.py" in ans
            and "README.md" not in ans and "config.json" not in ans
        ),
        expected_tokens=200,
    ),
    Task(
        name="find_todos",
        category="search",
        parallelizable=True,
        instruction=(
            "Find all TODO comments in .py files in the current directory. "
            "Use `grep -n 'TODO' *.py` to list them. "
            "Output each as 'filename: <todo text>', one per line."
        ),
        setup=lambda d: [
            _w(f"{d}/main.py", "# TODO: add error handling\ndef run(): pass\n# TODO: write tests\n"),
            _w(f"{d}/utils.py", "def helper():\n    pass  # TODO: implement this\n"),
            _w(f"{d}/clean.py", "def done(): return True\n"),
        ],
        validate=lambda d, ans: ans.count("TODO") >= 3,
        expected_tokens=250,
    ),
    Task(
        name="largest_file",
        category="search",
        parallelizable=True,
        instruction="Find the largest .py file by byte size. Use `ls -la *.py | sort -k5 -nr` to see sizes. Output just the filename, nothing else.",
        setup=lambda d: [
            _w(f"{d}/small.py", "x = 1\n"),
            _w(f"{d}/medium.py", "\n".join(f"x{i} = {i}" for i in range(50))),
            _w(f"{d}/large.py", "\n".join(f"x{i} = {i}" for i in range(200))),
        ],
        validate=lambda d, ans: "large.py" in ans,
        expected_tokens=180,
    ),
    Task(
        name="count_imports",
        category="search",
        parallelizable=True,
        instruction=(
            "Count how many import statements (lines starting with 'import' or 'from') "
            "exist across all .py files. Use `grep -E '^import|^from' *.py | wc -l`. Output just the number."
        ),
        setup=lambda d: [
            _w(f"{d}/a.py", "import os\nimport sys\ndef foo(): pass\n"),
            _w(f"{d}/b.py", "import json\nfrom pathlib import Path\n"),
            _w(f"{d}/c.py", "# no imports\ndef bar(): pass\n"),
        ],
        validate=lambda d, ans: "4" in ans,
        expected_tokens=200,
    ),
]

# ---------------------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------------------

TASKS += [
    Task(
        name="line_count_per_file",
        category="analysis",
        parallelizable=True,
        instruction="Count lines in each .py file. Output 'filename: N' for each, one per line.",
        setup=lambda d: [
            _w(f"{d}/alpha.py", "line1\nline2\nline3\n"),
            _w(f"{d}/beta.py", "a\nb\nc\nd\ne\n"),
        ],
        validate=lambda d, ans: (
            ("alpha" in ans and "3" in ans) and ("beta" in ans and "5" in ans)
        ),
        expected_tokens=200,
    ),
    Task(
        name="list_functions",
        category="analysis",
        parallelizable=True,
        instruction="List all Python function names defined across all .py files. One name per line.",
        setup=lambda d: [
            _w(f"{d}/module.py", "def alpha(): pass\ndef beta(x): return x\ndef gamma(a, b): return a+b\n"),
        ],
        validate=lambda d, ans: all(f in ans for f in ["alpha", "beta", "gamma"]),
        expected_tokens=200,
    ),
    Task(
        name="unique_imports",
        category="analysis",
        parallelizable=True,
        instruction="List all unique top-level module names imported across all .py files. One per line, sorted.",
        setup=lambda d: [
            _w(f"{d}/a.py", "import os\nimport sys\nimport json\n"),
            _w(f"{d}/b.py", "import os\nimport re\nfrom pathlib import Path\n"),
        ],
        validate=lambda d, ans: all(m in ans for m in ["json", "os", "pathlib", "re", "sys"]),
        expected_tokens=220,
    ),
    Task(
        name="total_line_count",
        category="analysis",
        parallelizable=True,
        instruction="Count the total number of lines across all .py files combined. Use `cat *.py | wc -l`. Output just the number.",
        setup=lambda d: [
            _w(f"{d}/a.py", "x\ny\nz\n"),          # 3
            _w(f"{d}/b.py", "a\nb\n"),              # 2
            _w(f"{d}/c.py", "1\n2\n3\n4\n5\n"),    # 5
        ],
        validate=lambda d, ans: "10" in ans,
        expected_tokens=180,
    ),
]

# ---------------------------------------------------------------------------
# TRANSFORM
# ---------------------------------------------------------------------------

TASKS += [
    Task(
        name="extract_todos_to_file",
        category="transform",
        parallelizable=False,
        instruction=(
            "Find all TODO comments in .py files and write them to todos.txt, one per line. "
            "Each line should contain only the TODO text (strip the # TODO: prefix)."
        ),
        setup=lambda d: [
            _w(f"{d}/app.py", "# TODO: fix login\ndef login(): pass\n# TODO: add tests\n"),
            _w(f"{d}/db.py", "# TODO: add index\ndef query(): pass\n"),
        ],
        validate=lambda d, ans: (
            os.path.exists(f"{d}/todos.txt")
            and len(_read(f"{d}/todos.txt").strip().splitlines()) >= 3
        ),
        expected_tokens=280,
    ),
    Task(
        name="file_sizes_report",
        category="transform",
        parallelizable=False,
        instruction=(
            "Write sizes.txt listing each .py file and its size in bytes. "
            "Format: 'filename bytes', one per line, sorted by size descending."
        ),
        setup=lambda d: [
            _w(f"{d}/big.py", "x = " + "1" * 200 + "\n"),
            _w(f"{d}/small.py", "x = 1\n"),
            _w(f"{d}/medium.py", "x = " + "1" * 80 + "\n"),
        ],
        validate=lambda d, ans: (
            os.path.exists(f"{d}/sizes.txt")
            and _read(f"{d}/sizes.txt").index("big.py") < _read(f"{d}/sizes.txt").index("small.py")
        ),
        expected_tokens=280,
    ),
    Task(
        name="rename_extension",
        category="transform",
        parallelizable=False,
        instruction="Rename all .txt files in the current directory to .md (same name, different extension).",
        setup=lambda d: [
            _w(f"{d}/notes.txt", "some notes\n"),
            _w(f"{d}/readme.txt", "readme content\n"),
            _w(f"{d}/keep.py", "x = 1\n"),
        ],
        validate=lambda d, ans: (
            os.path.exists(f"{d}/notes.md")
            and os.path.exists(f"{d}/readme.md")
            and not os.path.exists(f"{d}/notes.txt")
            and os.path.exists(f"{d}/keep.py")
        ),
        expected_tokens=240,
    ),
]

# ---------------------------------------------------------------------------
# SCRIPT WRITING
# ---------------------------------------------------------------------------

TASKS += [
    Task(
        name="script_sum_numbers",
        category="scripting",
        parallelizable=True,
        instruction=(
            "Write a Python script sum_file.py that opens 'numbers.txt' directly (no arguments) "
            "reads one integer per line, and prints their sum. "
            "Use cat > sum_file.py << 'EOF' to write it, then run python3 sum_file.py to verify it prints 100."
        ),
        setup=lambda d: [
            _w(f"{d}/numbers.txt", "10\n20\n30\n40\n"),
        ],
        validate=lambda d, ans: (
            os.path.exists(f"{d}/sum_file.py")
            and _run("python3 sum_file.py", d) == "100"
        ),
        expected_tokens=400,
    ),
    Task(
        name="script_count_unique_words",
        category="scripting",
        parallelizable=True,
        instruction=(
            "Write a Python script word_count.py that reads input.txt and prints "
            "the number of unique words (case-insensitive). "
            "Use cat > word_count.py << 'EOF' to write it, then verify with python3 word_count.py."
        ),
        setup=lambda d: [
            _w(f"{d}/input.txt", "the cat sat on the mat\nthe cat is fat\n"),
        ],
        validate=lambda d, ans: (
            os.path.exists(f"{d}/word_count.py")
            and _run("python3 word_count.py", d) == "7"
        ),
        expected_tokens=400,
    ),
    Task(
        name="script_filter_json",
        category="scripting",
        parallelizable=True,
        instruction=(
            "Write a Python script filter_json.py that reads data.json (a list of objects "
            "with 'name' and 'value' fields) and prints the names of objects where value > 50. "
            "Use cat > filter_json.py << 'EOF' to write it, then verify with python3 filter_json.py."
        ),
        setup=lambda d: [
            _w(f"{d}/data.json", json.dumps([
                {"name": "alpha", "value": 30},
                {"name": "beta", "value": 70},
                {"name": "gamma", "value": 90},
                {"name": "delta", "value": 10},
            ])),
        ],
        validate=lambda d, ans: (
            os.path.exists(f"{d}/filter_json.py")
            and (out := _run("python3 filter_json.py", d))
            and "beta" in out and "gamma" in out
            and "alpha" not in out and "delta" not in out
        ),
        expected_tokens=450,
    ),
    Task(
        name="script_find_duplicates",
        category="scripting",
        parallelizable=True,
        instruction=(
            "Write a Python script find_dupes.py that reads lines.txt and prints "
            "any duplicate lines (lines that appear more than once). "
            "Use cat > find_dupes.py << 'EOF' to write it, then verify with python3 find_dupes.py."
        ),
        setup=lambda d: [
            _w(f"{d}/lines.txt", "apple\nbanana\napple\ncherry\nbanana\ndate\n"),
        ],
        validate=lambda d, ans: (
            os.path.exists(f"{d}/find_dupes.py")
            and (out := _run("python3 find_dupes.py", d))
            and "apple" in out and "banana" in out and "cherry" not in out
        ),
        expected_tokens=400,
    ),
]

# ---------------------------------------------------------------------------
# MULTI-STEP
# ---------------------------------------------------------------------------

TASKS += [
    Task(
        name="most_todos_file",
        category="multi_step",
        parallelizable=False,
        instruction=(
            "Find which .py file has the most TODO comments. "
            "Run `grep -c TODO *.py` to get counts per file, then identify the highest manually. "
            "Report in the format: 'filename: N TODOs'."
        ),
        setup=lambda d: [
            _w(f"{d}/busy.py", "# TODO: a\n# TODO: b\n# TODO: c\ndef f(): pass\n"),
            _w(f"{d}/some.py", "# TODO: one thing\ndef g(): pass\n"),
            _w(f"{d}/clean.py", "def h(): pass\n"),
        ],
        validate=lambda d, ans: "busy.py" in ans and "3" in ans,
        expected_tokens=350,
    ),
    Task(
        name="largest_function_file",
        category="multi_step",
        parallelizable=False,
        instruction=(
            "Find which .py file defines the most functions. "
            "Use `grep -c '^def ' *.py` to count per file, then report the filename with the highest count "
            "in the format: 'filename: N functions'."
        ),
        setup=lambda d: [
            _w(f"{d}/rich.py", "def f1(): pass\ndef f2(): pass\ndef f3(): pass\n"),
            _w(f"{d}/poor.py", "def g1(): pass\ndef g2(): pass\n"),
            _w(f"{d}/empty.py", "x = 1\n"),
        ],
        validate=lambda d, ans: "rich.py" in ans and "3" in ans,
        expected_tokens=350,
    ),
    Task(
        name="find_and_report_errors",
        category="multi_step",
        parallelizable=False,
        instruction=(
            "Check each .py file for syntax errors using python3 -m py_compile. "
            "Report which files have errors and which are clean."
        ),
        setup=lambda d: [
            _w(f"{d}/good.py", "def f(): return 1\n"),
            _w(f"{d}/bad.py", "def broken(\n    pass\n"),
            _w(f"{d}/also_good.py", "x = [1, 2, 3]\n"),
        ],
        validate=lambda d, ans: "bad.py" in ans and "good.py" in ans,
        expected_tokens=400,
    ),
]

# ---------------------------------------------------------------------------
# TREE TASKS — each root decomposes into self-contained leaves
# ---------------------------------------------------------------------------
# Tree: "codebase_health_report"
# ├── tree_leaf: count_all_lines        (lines of code)
# ├── tree_leaf: count_all_functions    (function inventory)
# ├── tree_leaf: collect_all_todos      (tech debt)
# └── tree_leaf: check_all_syntax       (correctness)
# Root synthesizes the four leaf results into a report.

def _project_setup(d: str) -> None:
    _w(f"{d}/app.py", (
        "# TODO: add auth\nimport os\nimport json\n\n"
        "def start(): pass\ndef stop(): pass\n"
    ))
    _w(f"{d}/db.py", (
        "import json\n\n"
        "def connect(): pass\ndef query(): pass\ndef close(): pass\n"
        "# TODO: add connection pooling\n"
    ))
    _w(f"{d}/utils.py", (
        "import os\nimport re\n\n"
        "def parse(s): return s.strip()\n"
        "# TODO: handle unicode\n"
    ))
    _w(f"{d}/broken.py", "def oops(\n    pass\n")  # syntax error


TASKS += [
    Task(
        name="tree_count_lines",
        category="tree_leaf",
        parallelizable=True,
        tree_parent="codebase_health_report",
        instruction="Count the total number of lines across all .py files. Use `cat *.py | wc -l`. Output just the number.",
        setup=_project_setup,
        validate=lambda d, ans: any(n in ans for n in ["17", "18", "19"]),  # rough
        expected_tokens=200,
    ),
    Task(
        name="tree_list_functions",
        category="tree_leaf",
        parallelizable=True,
        tree_parent="codebase_health_report",
        instruction="List all function names defined across all .py files. One per line.",
        setup=_project_setup,
        validate=lambda d, ans: all(f in ans for f in ["start", "stop", "connect", "query", "close", "parse"]),
        expected_tokens=250,
    ),
    Task(
        name="tree_collect_todos",
        category="tree_leaf",
        parallelizable=True,
        tree_parent="codebase_health_report",
        instruction="Find all TODO comments in .py files. Use `grep -n 'TODO' *.py` then format as 'filename: <todo text>' for each.",
        setup=_project_setup,
        validate=lambda d, ans: ans.count("TODO") >= 3,
        expected_tokens=250,
    ),
    Task(
        name="tree_check_syntax",
        category="tree_leaf",
        parallelizable=True,
        tree_parent="codebase_health_report",
        instruction=(
            "Check each .py file for syntax errors using python3 -m py_compile. "
            "Output 'OK: filename' or 'ERROR: filename' for each file."
        ),
        setup=_project_setup,
        validate=lambda d, ans: "ERROR" in ans and "broken.py" in ans,
        expected_tokens=300,
    ),
    # Root task — receives pre-computed leaf results inline (simulates orchestrator passing context)
    Task(
        name="tree_root_health_report",
        category="tree_root",
        parallelizable=False,
        tree_parent=None,
        instruction=(
            "You have received the following analysis results from sub-agents:\n\n"
            "LINES: 18\n\n"
            "FUNCTIONS: start, stop, connect, query, close, parse\n\n"
            "TODOs:\n  app.py: add auth\n  db.py: add connection pooling\n  utils.py: handle unicode\n\n"
            "SYNTAX:\n  OK: app.py\n  OK: db.py\n  OK: utils.py\n  ERROR: broken.py\n\n"
            "Write a brief codebase health report to report.md using `cat > report.md << 'EOF'`. "
            "Summarise the line count, functions, TODOs, and syntax errors."
        ),
        setup=lambda d: None,
        validate=lambda d, ans: (
            os.path.exists(f"{d}/report.md")
            and "broken.py" in _read(f"{d}/report.md")
            and "TODO" in _read(f"{d}/report.md")
        ),
        expected_tokens=500,
    ),
]

# ---------------------------------------------------------------------------
# SEMANTIC SEARCH — model reads code and finds patterns no regex can express
# ---------------------------------------------------------------------------

TASKS += [
    Task(
        name="sem_missing_error_handling",
        category="semantic_search",
        parallelizable=True,
        instruction=(
            "Run `cat *.py` to read all Python files, then identify functions that open files or make "
            "network calls but don't handle exceptions. List each function name and why it's risky."
        ),
        setup=lambda d: [
            _w(f"{d}/io.py", (
                "import urllib.request\n\n"
                "def fetch(url):\n"
                "    return urllib.request.urlopen(url).read()\n\n"  # no try/except
                "def fetch_safe(url):\n"
                "    try:\n"
                "        return urllib.request.urlopen(url).read()\n"
                "    except Exception as e:\n"
                "        return None\n\n"
                "def read_config(path):\n"
                "    with open(path) as f:\n"  # no try/except
                "        return f.read()\n\n"
                "def read_config_safe(path):\n"
                "    try:\n"
                "        with open(path) as f:\n"
                "            return f.read()\n"
                "    except FileNotFoundError:\n"
                "        return None\n"
            )),
        ],
        validate=lambda d, ans: (
            "fetch" in ans and "read_config" in ans
            and "fetch_safe" not in ans.replace("fetch_safe", "")  # fetch_safe should not be flagged
        ),
        expected_tokens=600,
    ),
    Task(
        name="sem_duplicate_logic",
        category="semantic_search",
        parallelizable=True,
        instruction=(
            "Run `cat *.py` to read all Python files, then find functions that appear to implement "
            "the same logic in different ways. List the function pairs and explain the duplication."
        ),
        setup=lambda d: [
            _w(f"{d}/utils.py", (
                "def get_initials_v1(name):\n"
                "    parts = name.split()\n"
                "    return ''.join(p[0].upper() for p in parts)\n\n"
                "def make_abbreviation(full_name):\n"
                "    words = full_name.split(' ')\n"
                "    return ''.join([w[0].upper() for w in words])\n\n"
                "def days_since(date_str):\n"
                "    from datetime import datetime\n"
                "    d = datetime.strptime(date_str, '%Y-%m-%d')\n"
                "    return (datetime.now() - d).days\n\n"
                "def totally_unrelated(x, y):\n"
                "    return x * y + 1\n"
            )),
        ],
        validate=lambda d, ans: (
            "initials" in ans.lower() or "abbreviation" in ans.lower() or
            ("get_initials" in ans and "make_abbreviation" in ans)
        ),
        expected_tokens=700,
    ),
    Task(
        name="sem_off_by_one",
        category="semantic_search",
        parallelizable=True,
        instruction=(
            "Run `cat *.py` to read all Python files, then identify functions that likely have "
            "off-by-one errors in their loop bounds or index access. List each suspect function and explain."
        ),
        setup=lambda d: [
            _w(f"{d}/loops.py", (
                "def first_n(items, n):\n"
                "    result = []\n"
                "    for i in range(1, n + 1):  # starts at 1, skips first element\n"
                "        result.append(items[i])\n"
                "    return result\n\n"
                "def last_item(items):\n"
                "    return items[len(items)]  # should be len-1\n\n"
                "def middle(items):\n"
                "    return items[len(items) // 2]\n\n"  # correct
                "def chunk(items, size):\n"
                "    return [items[i:i+size] for i in range(0, len(items), size)]\n"  # correct
            )),
        ],
        validate=lambda d, ans: "first_n" in ans or "last_item" in ans,
        expected_tokens=700,
    ),
    Task(
        name="sem_missing_input_validation",
        category="semantic_search",
        parallelizable=True,
        instruction=(
            "Run `cat *.py` to read all Python files, then find functions that accept user-facing inputs "
            "but perform no validation before using them. List each function and what's missing."
        ),
        setup=lambda d: [
            _w(f"{d}/api.py", (
                "def create_user(username, age, email):\n"
                "    # no validation at all\n"
                "    db.insert({'username': username, 'age': age, 'email': email})\n\n"
                "def set_age(user_id, age):\n"
                "    if not isinstance(age, int) or age < 0 or age > 150:\n"
                "        raise ValueError('invalid age')\n"
                "    db.update(user_id, age=age)\n\n"
                "def delete_user(user_id):\n"
                "    if not user_id:\n"
                "        raise ValueError('user_id required')\n"
                "    db.delete(user_id)\n\n"
                "def rename(user_id, new_name):\n"
                "    # no check that new_name is non-empty or valid\n"
                "    db.update(user_id, name=new_name)\n"
            )),
        ],
        validate=lambda d, ans: "create_user" in ans and "rename" in ans,
        expected_tokens=700,
    ),
    Task(
        name="sem_dead_code",
        category="semantic_search",
        parallelizable=True,
        instruction=(
            "Run `cat *.py` to read all Python files, then identify functions or code blocks that "
            "appear to be unreachable or never called. Explain why each is dead code."
        ),
        setup=lambda d: [
            _w(f"{d}/code.py", (
                "def helper():\n"
                "    return 42\n\n"
                "def active():\n"
                "    x = helper()\n"
                "    return x * 2\n\n"
                "def legacy_process():\n"
                "    # replaced by active(), never called\n"
                "    return helper() + 1\n\n"
                "def always_returns_early(x):\n"
                "    if True:\n"
                "        return x\n"
                "    return x * 2  # unreachable\n\n"
                "result = active()\n"
            )),
        ],
        validate=lambda d, ans: "legacy_process" in ans or "always_returns_early" in ans,
        expected_tokens=700,
    ),
    Task(
        name="sem_wrong_abstraction",
        category="semantic_search",
        parallelizable=True,
        instruction=(
            "Run `cat *.py` to read all Python files, then identify functions that do more than one thing "
            "(violate single responsibility). List each and describe the multiple responsibilities."
        ),
        setup=lambda d: [
            _w(f"{d}/service.py", (
                "import json, smtplib\n\n"
                "def process_order(order_data):\n"
                "    # validates, saves to db, sends email, updates inventory — all in one\n"
                "    if not order_data.get('item'):\n"
                "        raise ValueError('missing item')\n"
                "    db.save(order_data)\n"
                "    smtp = smtplib.SMTP('localhost')\n"
                "    smtp.sendmail('shop@example.com', order_data['email'], 'Order confirmed')\n"
                "    inventory[order_data['item']] -= 1\n"
                "    return True\n\n"
                "def get_username(user_id):\n"
                "    return db.find(user_id)['name']\n\n"
                "def validate_email(email):\n"
                "    return '@' in email and '.' in email\n"
            )),
        ],
        validate=lambda d, ans: "process_order" in ans,
        expected_tokens=700,
    ),
]

# ---------------------------------------------------------------------------
# ITERATIVE — write → run → fail → fix → re-run (tight loop, Merlin's sweet spot)
# ---------------------------------------------------------------------------

TASKS += [
    Task(
        name="iter_write_and_test",
        category="iterative",
        parallelizable=False,
        instruction=(
            "Write a function `flatten(lst)` in flatten.py using `cat > flatten.py << 'EOF'` "
            "that recursively flattens a nested list. "
            "Then run `python3 test_flatten.py` to verify. Fix and re-run until all tests pass."
        ),
        setup=lambda d: [
            _w(f"{d}/test_flatten.py", (
                "from flatten import flatten\n"
                "assert flatten([1, [2, 3], [4, [5, 6]]]) == [1, 2, 3, 4, 5, 6]\n"
                "assert flatten([]) == []\n"
                "assert flatten([[1], [2], [3]]) == [1, 2, 3]\n"
                "assert flatten([1, [2, [3, [4]]]]) == [1, 2, 3, 4]\n"
                "print('all tests passed')\n"
            )),
        ],
        validate=lambda d, ans: (
            os.path.exists(f"{d}/flatten.py")
            and _run("python3 test_flatten.py", d) == "all tests passed"
        ),
        expected_tokens=800,
    ),
    Task(
        name="iter_debug_broken_script",
        category="iterative",
        parallelizable=False,
        instruction=(
            "Run `python3 process.py` to see the error. "
            "Then `cat process.py` to read it. Fix the bug using cat > process.py << 'EOF'. "
            "Run again to verify it prints 'done' with no exceptions."
        ),
        setup=lambda d: [
            _w(f"{d}/data.json", json.dumps([1, 2, 3, 4, 5])),
            _w(f"{d}/process.py", (
                "import json\n\n"
                "with open('data.json') as f:\n"
                "    items = json.load(f)\n\n"
                "total = 0\n"
                "for i in range(len(items) + 1):  # off-by-one: will IndexError\n"
                "    total += items[i]\n\n"
                "print(f'sum={total}')\n"
                "print('done')\n"
            )),
        ],
        validate=lambda d, ans: _run("python3 process.py", d).endswith("done"),
        expected_tokens=700,
    ),
    Task(
        name="iter_regex_refine",
        category="iterative",
        parallelizable=False,
        instruction=(
            "Write a regex pattern in pattern.txt using `echo 'your_pattern' > pattern.txt`. "
            "The pattern must match valid email addresses. "
            "Test it by running `python3 test_regex.py` and fix until all cases pass."
        ),
        setup=lambda d: [
            _w(f"{d}/test_regex.py", (
                "import re\n"
                "pattern = open('pattern.txt').read().strip()\n"
                "should_match = ['user@example.com', 'a.b@c.org', 'x+y@domain.co.uk']\n"
                "should_not = ['notanemail', '@nodomain', 'missing@', 'spaces in@email.com']\n"
                "for s in should_match:\n"
                "    assert re.fullmatch(pattern, s), f'should match: {s}'\n"
                "for s in should_not:\n"
                "    assert not re.fullmatch(pattern, s), f'should not match: {s}'\n"
                "print('all tests passed')\n"
            )),
        ],
        validate=lambda d, ans: (
            os.path.exists(f"{d}/pattern.txt")
            and _run("python3 test_regex.py", d) == "all tests passed"
        ),
        expected_tokens=750,
    ),
    Task(
        name="iter_implement_to_spec",
        category="iterative",
        parallelizable=False,
        instruction=(
            "Implement `run_length_encode(s)` in rle.py using `cat > rle.py << 'EOF'`. "
            "It should return a list of (char, count) tuples. "
            "Run `python3 test_rle.py` and fix until all tests pass."
        ),
        setup=lambda d: [
            _w(f"{d}/test_rle.py", (
                "from rle import run_length_encode as rle\n"
                "assert rle('aabbbcc') == [('a',2),('b',3),('c',2)]\n"
                "assert rle('abc') == [('a',1),('b',1),('c',1)]\n"
                "assert rle('') == []\n"
                "assert rle('aaaa') == [('a',4)]\n"
                "print('all tests passed')\n"
            )),
        ],
        validate=lambda d, ans: (
            os.path.exists(f"{d}/rle.py")
            and _run("python3 test_rle.py", d) == "all tests passed"
        ),
        expected_tokens=700,
    ),
]

# ---------------------------------------------------------------------------
# FILESYSTEM PIPELINE — intermediate files as working memory between steps
# ---------------------------------------------------------------------------

TASKS += [
    Task(
        name="pipeline_extract_and_report",
        category="pipeline",
        parallelizable=False,
        instruction=(
            "Write a Python script process.py that reads metrics.json (list of {name, value} objects), "
            "writes passing.txt with names where value >= 80, failing.txt with names where value < 80, "
            "and summary.txt with the count of each. "
            "Use cat > process.py << 'EOF' to write it, then run python3 process.py."
        ),
        setup=lambda d: [
            _w(f"{d}/metrics.json", json.dumps([
                {"name": "auth", "value": 95},
                {"name": "db", "value": 72},
                {"name": "api", "value": 88},
                {"name": "cache", "value": 61},
                {"name": "worker", "value": 83},
            ])),
        ],
        validate=lambda d, ans: (
            os.path.exists(f"{d}/passing.txt") and os.path.exists(f"{d}/failing.txt")
            and "auth" in _read(f"{d}/passing.txt")
            and "db" in _read(f"{d}/failing.txt")
            and os.path.exists(f"{d}/summary.txt")
        ),
        expected_tokens=500,
    ),
    Task(
        name="pipeline_merge_configs",
        category="pipeline",
        parallelizable=False,
        instruction=(
            "Write a Python script merge.py that merges base.json, env.json, and local.json into merged.json. "
            "Keys in local.json override env.json, which overrides base.json. "
            "Use cat > merge.py << 'EOF' to write it, then run python3 merge.py to execute."
        ),
        setup=lambda d: [
            _w(f"{d}/base.json",  json.dumps({"host": "localhost", "port": 5432, "debug": False, "timeout": 30})),
            _w(f"{d}/env.json",   json.dumps({"host": "prod.db.internal", "debug": False})),
            _w(f"{d}/local.json", json.dumps({"debug": True, "timeout": 5})),
        ],
        validate=lambda d, ans: (
            os.path.exists(f"{d}/merged.json")
            and (cfg := json.loads(_read(f"{d}/merged.json")))
            and cfg.get("host") == "prod.db.internal"
            and cfg.get("debug") is True
            and cfg.get("timeout") == 5
            and cfg.get("port") == 5432
        ),
        expected_tokens=500,
    ),
    Task(
        name="pipeline_csv_aggregate",
        category="pipeline",
        parallelizable=False,
        instruction=(
            "Write a Python script agg.py that reads sales.csv (columns: region, product, amount) "
            "and writes region_totals.csv with total amount per region, sorted by total descending. "
            "Use cat > agg.py << 'EOF' to write it, then run python3 agg.py."
        ),
        setup=lambda d: [
            _w(f"{d}/sales.csv", (
                "region,product,amount\n"
                "north,widget,100\nnorth,gadget,200\n"
                "south,widget,150\nsouth,gadget,50\n"
                "east,widget,300\n"
                "north,widget,50\n"
            )),
        ],
        validate=lambda d, ans: (
            os.path.exists(f"{d}/region_totals.csv")
            and (content := _read(f"{d}/region_totals.csv"))
            and content.index("north") < content.index("east")  # north=350 > east=300, north appears first
        ),
        expected_tokens=600,
    ),
    Task(
        name="pipeline_build_index",
        category="pipeline",
        parallelizable=False,
        instruction=(
            "Write a Python script index.py that reads all .py files and builds index.json: "
            "a dict mapping each function name to the file it's defined in. "
            "Use cat > index.py << 'EOF' to write it, then run python3 index.py."
        ),
        setup=lambda d: [
            _w(f"{d}/auth.py",   "def login(): pass\ndef logout(): pass\ndef refresh(): pass\n"),
            _w(f"{d}/db.py",     "def connect(): pass\ndef query(): pass\n"),
            _w(f"{d}/utils.py",  "def parse(): pass\ndef format(): pass\ndef validate(): pass\n"),
        ],
        validate=lambda d, ans: (
            os.path.exists(f"{d}/index.json")
            and (idx := json.loads(_read(f"{d}/index.json")))
            and idx.get("login") and "auth" in idx["login"]
            and idx.get("connect") and "db" in idx["connect"]
        ),
        expected_tokens=500,
    ),
]

# ---------------------------------------------------------------------------
# PARALLEL BATCH — 8 independent modules, same task, validates spawn model
# ---------------------------------------------------------------------------

_modules = [
    ("auth",    ["login", "logout", "verify"],          "import hashlib\n"),
    ("db",      ["connect", "query", "close"],          "import sqlite3\n"),
    ("cache",   ["get", "set", "delete"],               "import redis\n"),
    ("api",     ["get_user", "create_user"],            "import flask\n"),
    ("worker",  ["start", "stop", "enqueue"],           "import celery\n"),
    ("mailer",  ["send", "render_template"],            "import smtplib\n"),
    ("storage", ["upload", "download", "delete_file"],  "import boto3\n"),
    ("metrics", ["record", "flush"],                    "import statsd\n"),
]

for mod, funcs, imp in _modules:
    body = imp + "\n" + "\n".join(f"def {f}(): pass" for f in funcs) + "\n"
    TASKS.append(Task(
        name=f"parallel_list_functions_{mod}",
        category="parallel",
        parallelizable=True,
        instruction=f"List all function names defined in {mod}.py. One per line.",
        setup=lambda d, m=mod, b=body: _w(f"{d}/{m}.py", b),
        validate=lambda d, ans, fs=funcs: all(f in ans for f in fs),
        expected_tokens=200,
    ))

# ---------------------------------------------------------------------------
# DATASET CLEANING — filter, transform, deduplicate structured data
# ---------------------------------------------------------------------------

TASKS += [
    Task(
        name="dataset_filter_short",
        category="pipeline",
        parallelizable=True,
        instruction=(
            "Write a Python script clean.py using `cat > clean.py << 'EOF'` that reads records.jsonl "
            "(one JSON object per line), filters out records where the 'text' field has fewer than 20 characters, "
            "and writes the passing records to filtered.jsonl. "
            "Run `python3 clean.py` to execute."
        ),
        setup=lambda d: _w(f"{d}/records.jsonl", "\n".join(json.dumps(r) for r in [
            {"id": 1, "text": "too short"},
            {"id": 2, "text": "this is a longer piece of text that should pass the filter"},
            {"id": 3, "text": "hi"},
            {"id": 4, "text": "another record that is definitely long enough to pass"},
            {"id": 5, "text": "ok"},
        ])),
        validate=lambda d, ans: (
            os.path.exists(f"{d}/filtered.jsonl")
            and (lines := _read(f"{d}/filtered.jsonl").strip().splitlines())
            and len(lines) == 2
            and all(json.loads(l)["id"] in [2, 4] for l in lines)
        ),
        expected_tokens=500,
    ),
    Task(
        name="dataset_dedup",
        category="pipeline",
        parallelizable=True,
        instruction=(
            "Write a Python script dedup.py using `cat > dedup.py << 'EOF'` that reads items.jsonl "
            "(one JSON object per line with a 'text' field), removes duplicate texts (keep first occurrence), "
            "and writes deduped.jsonl with the unique records. "
            "Run `python3 dedup.py` to execute."
        ),
        setup=lambda d: _w(f"{d}/items.jsonl", "\n".join(json.dumps(r) for r in [
            {"id": 1, "text": "hello world"},
            {"id": 2, "text": "foo bar"},
            {"id": 3, "text": "hello world"},
            {"id": 4, "text": "baz qux"},
            {"id": 5, "text": "foo bar"},
        ])),
        validate=lambda d, ans: (
            os.path.exists(f"{d}/deduped.jsonl")
            and (lines := _read(f"{d}/deduped.jsonl").strip().splitlines())
            and len(lines) == 3
            and {json.loads(l)["id"] for l in lines} == {1, 2, 4}
        ),
        expected_tokens=500,
    ),
    Task(
        name="dataset_normalize",
        category="pipeline",
        parallelizable=True,
        instruction=(
            "Write a Python script normalize.py using `cat > normalize.py << 'EOF'` that reads users.jsonl "
            "(one JSON object per line), normalizes each record by: lowercasing the 'email' field, "
            "stripping whitespace from 'name', and writes the normalized records to clean_users.jsonl. "
            "Run `python3 normalize.py` to execute."
        ),
        setup=lambda d: _w(f"{d}/users.jsonl", "\n".join(json.dumps(r) for r in [
            {"id": 1, "name": "  Alice  ", "email": "ALICE@Example.COM"},
            {"id": 2, "name": "Bob", "email": "Bob@TEST.org"},
            {"id": 3, "name": " Charlie ", "email": "CHARLIE@DOMAIN.NET"},
        ])),
        validate=lambda d, ans: (
            os.path.exists(f"{d}/clean_users.jsonl")
            and (lines := _read(f"{d}/clean_users.jsonl").strip().splitlines())
            and len(lines) == 3
            and json.loads(lines[0])["email"] == "alice@example.com"
            and json.loads(lines[0])["name"] == "Alice"
        ),
        expected_tokens=500,
    ),
    Task(
        name="dataset_split_train_val",
        category="pipeline",
        parallelizable=True,
        instruction=(
            "Write a Python script split.py using `cat > split.py << 'EOF'` that reads corpus.jsonl "
            "(one JSON object per line), takes the first 80% of records as train.jsonl and the remaining "
            "20% as val.jsonl (no shuffling). Run `python3 split.py` to execute."
        ),
        setup=lambda d: _w(f"{d}/corpus.jsonl", "\n".join(
            json.dumps({"id": i, "text": f"record {i}"}) for i in range(10)
        )),
        validate=lambda d, ans: (
            os.path.exists(f"{d}/train.jsonl") and os.path.exists(f"{d}/val.jsonl")
            and len(_read(f"{d}/train.jsonl").strip().splitlines()) == 8
            and len(_read(f"{d}/val.jsonl").strip().splitlines()) == 2
        ),
        expected_tokens=500,
    ),
]
