"""
AST-based code mutation engine.

Applies a single controlled mutation to a Python source file, producing a
broken version that causes at least one test to fail. The failing tests
become the task specification; passing them becomes the agent's goal.

Mutation types:
  - swap_operator     — change a binary operator (+→-, *→/, etc.)
  - swap_compare      — change a comparison operator (==→!=, >→<, etc.)
  - negate_condition  — negate an if/while condition
  - delete_statement  — remove a statement from a function body
  - change_constant   — increment/decrement an integer constant
  - delete_return     — replace return expr with return None
"""

import ast
import copy
import random
import textwrap
from dataclasses import dataclass
from typing import Callable


# ---------------------------------------------------------------------------
# Mutation descriptors
# ---------------------------------------------------------------------------

@dataclass
class Mutation:
    kind: str
    description: str        # human-readable, used in task instruction
    original: str           # original source
    mutated: str            # mutated source
    lineno: int             # line where mutation was applied


# ---------------------------------------------------------------------------
# Node finders
# ---------------------------------------------------------------------------

def _find_nodes(tree: ast.AST, pred: Callable) -> list[ast.AST]:
    return [node for node in ast.walk(tree) if pred(node)]


def _in_function(tree: ast.AST, target: ast.AST) -> bool:
    """Check if a node is inside a function definition."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if target in ast.walk(node):
                return True
    return False


# ---------------------------------------------------------------------------
# Mutation implementations
# ---------------------------------------------------------------------------

BINARY_OP_SWAPS = {
    ast.Add: ast.Sub, ast.Sub: ast.Add,
    ast.Mult: ast.Div, ast.Div: ast.Mult,
    ast.FloorDiv: ast.Mod, ast.Mod: ast.FloorDiv,
    ast.BitAnd: ast.BitOr, ast.BitOr: ast.BitAnd,
    ast.LShift: ast.RShift, ast.RShift: ast.LShift,
}

COMPARE_OP_SWAPS = {
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
    ast.Lt: ast.GtE, ast.GtE: ast.Lt,
    ast.Gt: ast.LtE, ast.LtE: ast.Gt,
    ast.Is: ast.IsNot, ast.IsNot: ast.Is,
    ast.In: ast.NotIn, ast.NotIn: ast.In,
}


def _apply_swap_operator(tree: ast.AST, rng: random.Random) -> tuple[ast.AST, str, int] | None:
    candidates = [
        n for n in _find_nodes(tree, lambda n: isinstance(n, ast.BinOp))
        if type(n.op) in BINARY_OP_SWAPS
    ]
    if not candidates:
        return None
    node = rng.choice(candidates)
    original_op = type(node.op).__name__
    node.op = BINARY_OP_SWAPS[type(node.op)]()
    new_op = type(node.op).__name__
    return tree, f"swapped binary operator {original_op} → {new_op}", node.lineno


def _apply_swap_compare(tree: ast.AST, rng: random.Random) -> tuple[ast.AST, str, int] | None:
    candidates = [
        n for n in _find_nodes(tree, lambda n: isinstance(n, ast.Compare))
        if n.ops and type(n.ops[0]) in COMPARE_OP_SWAPS
    ]
    if not candidates:
        return None
    node = rng.choice(candidates)
    original_op = type(node.ops[0]).__name__
    node.ops[0] = COMPARE_OP_SWAPS[type(node.ops[0])]()
    new_op = type(node.ops[0]).__name__
    return tree, f"swapped comparison {original_op} → {new_op}", node.lineno


def _apply_negate_condition(tree: ast.AST, rng: random.Random) -> tuple[ast.AST, str, int] | None:
    candidates = _find_nodes(tree, lambda n: isinstance(n, (ast.If, ast.While)))
    if not candidates:
        return None
    node = rng.choice(candidates)
    lineno = node.lineno
    node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
    ast.fix_missing_locations(node)
    return tree, "negated condition", lineno


def _apply_delete_statement(tree: ast.AST, rng: random.Random) -> tuple[ast.AST, str, int] | None:
    """Delete a non-trivial statement from a function body."""
    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            # Skip functions with only one statement (would leave empty body)
            if len(body) <= 1:
                continue
            for i, stmt in enumerate(body):
                # Skip docstrings and pass
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    continue
                if isinstance(stmt, ast.Pass):
                    continue
                candidates.append((body, i, stmt.lineno))
    if not candidates:
        return None
    body, idx, lineno = rng.choice(candidates)
    del body[idx]
    return tree, "deleted a statement from function body", lineno


def _apply_change_constant(tree: ast.AST, rng: random.Random) -> tuple[ast.AST, str, int] | None:
    candidates = [
        n for n in _find_nodes(tree, lambda n: isinstance(n, ast.Constant))
        if isinstance(n.value, int) and n.value not in (0, 1, -1, True, False)
    ]
    if not candidates:
        # Fall back to 0/1 constants
        candidates = [
            n for n in _find_nodes(tree, lambda n: isinstance(n, ast.Constant))
            if isinstance(n.value, int) and not isinstance(n.value, bool)
        ]
    if not candidates:
        return None
    node = rng.choice(candidates)
    original = node.value
    node.value = original + rng.choice([-1, 1])
    return tree, f"changed integer constant {original} → {node.value}", node.lineno


def _apply_delete_return(tree: ast.AST, rng: random.Random) -> tuple[ast.AST, str, int] | None:
    candidates = [
        n for n in _find_nodes(tree, lambda n: isinstance(n, ast.Return))
        if n.value is not None
    ]
    if not candidates:
        return None
    node = rng.choice(candidates)
    lineno = node.lineno
    node.value = ast.Constant(value=None)
    ast.fix_missing_locations(node)
    return tree, "replaced return value with None", lineno


# ---------------------------------------------------------------------------
# Mutator
# ---------------------------------------------------------------------------

MUTATION_FNS = [
    _apply_swap_operator,
    _apply_swap_compare,
    _apply_negate_condition,
    _apply_delete_statement,
    _apply_change_constant,
    _apply_delete_return,
]


def mutate(source: str, seed: int | None = None) -> Mutation | None:
    """
    Apply a single random mutation to Python source.
    Returns Mutation on success, None if no applicable mutation found.
    """
    rng = random.Random(seed)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    # Shuffle mutation types and try until one applies
    fns = MUTATION_FNS.copy()
    rng.shuffle(fns)

    for fn in fns:
        mutated_tree = copy.deepcopy(tree)
        result = fn(mutated_tree, rng)
        if result is None:
            continue
        _, description, lineno = result
        try:
            ast.fix_missing_locations(mutated_tree)
            mutated_source = ast.unparse(mutated_tree)
            # Verify the mutated source is valid Python
            ast.parse(mutated_source)
            return Mutation(
                kind=fn.__name__.removeprefix("_apply_"),
                description=description,
                original=source,
                mutated=mutated_source,
                lineno=lineno,
            )
        except Exception:
            continue

    return None
