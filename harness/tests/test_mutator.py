"""Tests for AST mutation engine."""

import ast
import pytest
from harness.mutator import (
    mutate, Mutation,
    _apply_swap_operator, _apply_swap_compare, _apply_negate_condition,
    _apply_delete_statement, _apply_change_constant, _apply_delete_return,
    MUTATION_FNS,
)
import random
import copy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse(source: str) -> ast.AST:
    return ast.parse(source)


def apply(fn, source: str, seed: int = 42):
    tree = parse(source)
    mutated = copy.deepcopy(tree)
    rng = random.Random(seed)
    return fn(mutated, rng)


# ---------------------------------------------------------------------------
# Per-mutation tests
# ---------------------------------------------------------------------------

def test_swap_operator_add_to_sub():
    source = "def f(x): return x + 1"
    result = apply(_apply_swap_operator, source)
    assert result is not None
    _, desc, _ = result
    assert "Sub" in desc or "Add" in desc


def test_swap_operator_no_binop():
    source = "def f(x): return x"
    result = apply(_apply_swap_operator, source)
    assert result is None


def test_swap_compare_eq_to_noteq():
    source = "def f(x): return x == 1"
    result = apply(_apply_swap_compare, source)
    assert result is not None
    _, desc, _ = result
    assert "NotEq" in desc or "Eq" in desc


def test_negate_condition():
    source = "def f(x):\n    if x > 0:\n        return 1\n    return 0"
    result = apply(_apply_negate_condition, source)
    assert result is not None
    tree, desc, _ = result
    mutated = ast.unparse(tree)
    assert "not" in mutated
    assert "negated" in desc


def test_delete_statement_removes_one():
    source = "def f(x):\n    y = x + 1\n    z = y * 2\n    return z"
    result = apply(_apply_delete_statement, source)
    assert result is not None
    tree, _, _ = result
    mutated = ast.unparse(tree)
    original_stmts = 3  # y=, z=, return
    parsed = ast.parse(mutated)
    func = next(n for n in ast.walk(parsed) if isinstance(n, ast.FunctionDef))
    assert len(func.body) == original_stmts - 1


def test_delete_statement_skips_single_body():
    source = "def f(x):\n    return x"
    result = apply(_apply_delete_statement, source)
    assert result is None  # only one statement, can't delete


def test_change_constant():
    source = "def f(): return 42"
    result = apply(_apply_change_constant, source)
    assert result is not None
    tree, desc, _ = result
    mutated = ast.unparse(tree)
    assert "42" not in mutated or "41" in mutated or "43" in mutated
    assert "42" in desc or "changed" in desc


def test_delete_return():
    source = "def f(x): return x + 1"
    result = apply(_apply_delete_return, source)
    assert result is not None
    tree, desc, _ = result
    mutated = ast.unparse(tree)
    assert "return None" in mutated
    assert "None" in desc


def test_delete_return_no_return_value():
    source = "def f(x): return"
    result = apply(_apply_delete_return, source)
    assert result is None  # no return value to delete


# ---------------------------------------------------------------------------
# Top-level mutate()
# ---------------------------------------------------------------------------

def test_mutate_returns_mutation():
    source = "def f(x):\n    if x > 0:\n        return x + 1\n    return 0"
    m = mutate(source, seed=42)
    assert isinstance(m, Mutation)
    assert m.original == source
    assert m.mutated != source
    assert m.kind
    assert m.description
    assert m.lineno > 0


def test_mutate_produces_valid_python():
    source = "def f(x):\n    y = x * 2\n    if y == 10:\n        return True\n    return False"
    m = mutate(source, seed=0)
    assert m is not None
    # Must parse without error
    ast.parse(m.mutated)


def test_mutate_deterministic():
    source = "def f(x):\n    return x + 1 if x > 0 else x - 1"
    m1 = mutate(source, seed=99)
    m2 = mutate(source, seed=99)
    assert m1.mutated == m2.mutated
    assert m1.kind == m2.kind


def test_mutate_different_seeds():
    source = (
        "def f(x):\n"
        "    if x > 0:\n"
        "        return x + 1\n"
        "    elif x == 0:\n"
        "        return x * 2\n"
        "    return x - 1\n"
    )
    results = {mutate(source, seed=i).mutated for i in range(20)}
    # Should produce at least 2 distinct mutations across 20 seeds
    assert len(results) >= 2


def test_mutate_invalid_source():
    assert mutate("def f(x: this is not python") is None


def test_mutate_trivial_source():
    # A file with no mutable nodes
    source = "x = 'hello'"
    # May return None or a mutation — just check it doesn't crash
    result = mutate(source, seed=0)
    if result is not None:
        ast.parse(result.mutated)
