"""
AST-based code mutation engine.

Applies a single controlled mutation to a Python source file, producing a
broken version that causes at least one test to fail. The failing tests
become the task specification; passing them becomes the agent's goal.

Mutation types:
  - swap_operator          — change a binary operator (+→-, *→/, etc.)
  - swap_compare           — change a comparison operator (==→!=, >→<, etc.)
  - swap_boolean_operator  — and → or, or → and
  - swap_augmented_assign  — += → -=, *= → //=, etc.
  - negate_condition       — negate an if/while condition
  - drop_condition         — replace if/while condition with True
  - swap_if_else_body      — swap the if branch and else branch
  - delete_statement       — remove a statement from a function body
  - change_constant        — increment/decrement an integer constant
  - change_subscript_index — off-by-one in a subscript integer index
  - off_by_one_slice       — add/subtract 1 from a slice bound
  - flip_boolean           — True → False, False → True
  - change_default_arg     — change a function default argument value
  - delete_return          — replace return expr with return None
  - remove_not             — remove `not` from `not expr`
  - wrong_attribute        — swap two attribute accesses on the same object
  - swap_arguments         — swap two adjacent positional arguments in a call
  - empty_collection       — replace a non-empty list/dict/set literal with empty
  - swap_exception_type    — ValueError → TypeError, etc.
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
# New mutation implementations
# ---------------------------------------------------------------------------

BOOL_OP_SWAPS = {ast.And: ast.Or, ast.Or: ast.And}
AUG_OP_SWAPS = {
    ast.Add: ast.Sub, ast.Sub: ast.Add,
    ast.Mult: ast.Div, ast.Div: ast.Mult,
    ast.FloorDiv: ast.Mod, ast.Mod: ast.FloorDiv,
}
_EXCEPTION_TYPES = [
    "ValueError", "TypeError", "KeyError", "IndexError", "AttributeError",
    "RuntimeError", "NotImplementedError", "OSError", "AssertionError",
]


def _apply_swap_boolean_operator(tree: ast.AST, rng: random.Random) -> tuple | None:
    candidates = [
        n for n in _find_nodes(tree, lambda n: isinstance(n, ast.BoolOp))
        if type(n.op) in BOOL_OP_SWAPS
    ]
    if not candidates:
        return None
    node = rng.choice(candidates)
    original = type(node.op).__name__
    node.op = BOOL_OP_SWAPS[type(node.op)]()
    return tree, f"swapped boolean operator {original} → {type(node.op).__name__}", node.lineno


def _apply_swap_augmented_assign(tree: ast.AST, rng: random.Random) -> tuple | None:
    candidates = [
        n for n in _find_nodes(tree, lambda n: isinstance(n, ast.AugAssign))
        if type(n.op) in AUG_OP_SWAPS
    ]
    if not candidates:
        return None
    node = rng.choice(candidates)
    original = type(node.op).__name__
    node.op = AUG_OP_SWAPS[type(node.op)]()
    return tree, f"swapped augmented assign {original} → {type(node.op).__name__}", node.lineno


def _apply_drop_condition(tree: ast.AST, rng: random.Random) -> tuple | None:
    """Replace if/while condition with True — body always executes."""
    candidates = _find_nodes(tree, lambda n: isinstance(n, (ast.If, ast.While)))
    if not candidates:
        return None
    node = rng.choice(candidates)
    lineno = node.lineno
    node.test = ast.Constant(value=True)
    ast.fix_missing_locations(node)
    return tree, "replaced condition with True", lineno


def _apply_swap_if_else_body(tree: ast.AST, rng: random.Random) -> tuple | None:
    """Swap if branch and else branch (skips elif chains)."""
    candidates = [
        n for n in _find_nodes(tree, lambda n: isinstance(n, ast.If))
        if n.orelse and not isinstance(n.orelse[0], ast.If)
    ]
    if not candidates:
        return None
    node = rng.choice(candidates)
    node.body, node.orelse = node.orelse, node.body
    return tree, "swapped if/else branches", node.lineno


def _apply_flip_boolean(tree: ast.AST, rng: random.Random) -> tuple | None:
    candidates = [
        n for n in _find_nodes(tree, lambda n: isinstance(n, ast.Constant))
        if isinstance(n.value, bool)
    ]
    if not candidates:
        return None
    node = rng.choice(candidates)
    original = node.value
    node.value = not node.value
    return tree, f"flipped boolean {original} → {node.value}", node.lineno


def _apply_change_subscript_index(tree: ast.AST, rng: random.Random) -> tuple | None:
    """Off-by-one in a subscript integer index: arr[0] → arr[1]."""
    candidates = [
        n for n in _find_nodes(tree, lambda n: isinstance(n, ast.Subscript))
        if isinstance(n.slice, ast.Constant)
        and isinstance(n.slice.value, int)
        and not isinstance(n.slice.value, bool)
    ]
    if not candidates:
        return None
    node = rng.choice(candidates)
    original = node.slice.value
    node.slice.value = original + rng.choice([-1, 1])
    return tree, f"changed subscript index {original} → {node.slice.value}", node.lineno


def _apply_off_by_one_slice(tree: ast.AST, rng: random.Random) -> tuple | None:
    """Add/subtract 1 from a slice bound: arr[1:] → arr[2:]."""
    candidates = []
    for n in _find_nodes(tree, lambda n: isinstance(n, ast.Subscript)):
        if isinstance(n.slice, ast.Slice):
            for bound in (n.slice.lower, n.slice.upper):
                if (isinstance(bound, ast.Constant)
                        and isinstance(bound.value, int)
                        and not isinstance(bound.value, bool)):
                    candidates.append((n.lineno, bound))
    if not candidates:
        return None
    lineno, node = rng.choice(candidates)
    original = node.value
    node.value = original + rng.choice([-1, 1])
    return tree, f"off-by-one in slice bound {original} → {node.value}", lineno


def _apply_change_default_arg(tree: ast.AST, rng: random.Random) -> tuple | None:
    """Change a function default argument: def f(x=5) → def f(x=4)."""
    candidates = []
    for n in _find_nodes(tree, lambda n: isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        for d in n.args.defaults:
            if isinstance(d, ast.Constant) and isinstance(d.value, (int, bool)):
                candidates.append(d)
    if not candidates:
        return None
    node = rng.choice(candidates)
    original = node.value
    if isinstance(original, bool):
        node.value = not original
    else:
        node.value = original + rng.choice([-1, 1])
    return tree, f"changed default arg {original} → {node.value}", node.lineno


def _apply_remove_not(tree: ast.AST, rng: random.Random) -> tuple | None:
    """Remove `not` from `not expr`."""
    candidates = [
        n for n in _find_nodes(tree, lambda n: isinstance(n, ast.UnaryOp))
        if isinstance(n.op, ast.Not)
    ]
    if not candidates:
        return None
    target = rng.choice(candidates)

    class _RemoveNot(ast.NodeTransformer):
        def visit_UnaryOp(self, node):
            if node is target:
                return ast.copy_location(self.generic_visit(node.operand), node)
            return self.generic_visit(node)

    new_tree = _RemoveNot().visit(tree)
    return new_tree, "removed `not` from expression", target.lineno


def _apply_wrong_attribute(tree: ast.AST, rng: random.Random) -> tuple | None:
    """Swap two attribute accesses on the same object: self.x → self.y."""
    from collections import defaultdict
    obj_attrs: dict = defaultdict(list)
    for n in _find_nodes(tree, lambda n: isinstance(n, ast.Attribute)):
        if isinstance(n.value, ast.Name):
            obj_attrs[n.value.id].append(n)
    # Need an object with ≥2 distinct attribute names
    candidates = [
        (obj, nodes)
        for obj, nodes in obj_attrs.items()
        if len({n.attr for n in nodes}) >= 2
    ]
    if not candidates:
        return None
    obj, nodes = rng.choice(candidates)
    attrs = list({n.attr for n in nodes})
    rng.shuffle(attrs)
    a, b = attrs[0], attrs[1]
    node = rng.choice([n for n in nodes if n.attr == a])
    node.attr = b
    return tree, f"changed .{a} → .{b} on {obj}", node.lineno


def _apply_swap_arguments(tree: ast.AST, rng: random.Random) -> tuple | None:
    """Swap two adjacent positional arguments in a function call."""
    candidates = [
        n for n in _find_nodes(tree, lambda n: isinstance(n, ast.Call))
        if len(n.args) >= 2
    ]
    if not candidates:
        return None
    node = rng.choice(candidates)
    idx = rng.randint(0, len(node.args) - 2)
    node.args[idx], node.args[idx + 1] = node.args[idx + 1], node.args[idx]
    return tree, f"swapped arguments {idx} and {idx + 1} in call", node.lineno


def _apply_empty_collection(tree: ast.AST, rng: random.Random) -> tuple | None:
    """Replace a non-empty list/dict/set literal with the empty equivalent."""
    candidates = [
        n for n in ast.walk(tree)
        if hasattr(n, "lineno") and (
            (isinstance(n, (ast.List, ast.Set)) and n.elts)
            or (isinstance(n, ast.Dict) and n.keys)
        )
    ]
    if not candidates:
        return None
    node = rng.choice(candidates)
    kind = type(node).__name__.lower()
    if isinstance(node, (ast.List, ast.Set)):
        node.elts = []
    else:
        node.keys = []
        node.values = []
    return tree, f"emptied {kind} literal", node.lineno


def _apply_swap_exception_type(tree: ast.AST, rng: random.Random) -> tuple | None:
    """Change exception type in a raise: ValueError → TypeError."""
    candidates = [
        n for n in _find_nodes(tree, lambda n: isinstance(n, ast.Raise))
        if n.exc is not None
    ]
    if not candidates:
        return None
    node = rng.choice(candidates)
    # Extract name node
    if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
        name_node = node.exc.func
    elif isinstance(node.exc, ast.Name):
        name_node = node.exc
    else:
        return None
    current = name_node.id
    others = [e for e in _EXCEPTION_TYPES if e != current]
    if not others:
        return None
    name_node.id = rng.choice(others)
    return tree, f"changed exception {current} → {name_node.id}", node.lineno


# ---------------------------------------------------------------------------
# Mutator
# ---------------------------------------------------------------------------

MUTATION_FNS = [
    _apply_swap_operator,
    _apply_swap_compare,
    _apply_swap_boolean_operator,
    _apply_swap_augmented_assign,
    _apply_negate_condition,
    _apply_drop_condition,
    _apply_swap_if_else_body,
    _apply_delete_statement,
    _apply_change_constant,
    _apply_change_subscript_index,
    _apply_off_by_one_slice,
    _apply_flip_boolean,
    _apply_change_default_arg,
    _apply_delete_return,
    _apply_remove_not,
    _apply_wrong_attribute,
    _apply_swap_arguments,
    _apply_empty_collection,
    _apply_swap_exception_type,
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
