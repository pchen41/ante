"""Priority expressions written by the planner.

The planner may steer a search by writing one arithmetic expression (FunSearch-style: the model writes
the scoring function, a fixed search skeleton uses it). The expression is parsed and checked against a
small whitelist before it is ever evaluated: numbers, the named variables, arithmetic, comparisons,
`a if c else b`, and a few math functions. No attribute access, no subscripts, no calls to anything
else, no names beyond the variables. A rejected expression falls back to the default.
"""

from __future__ import annotations

import ast
import math

FUNCS = {
    "abs": abs, "min": min, "max": max, "sqrt": lambda v: math.sqrt(max(v, 0.0)), "exp": lambda v: math.exp(min(v, 50.0)),
    "log": lambda v: math.log(max(v, 1e-12)), "sin": math.sin, "cos": math.cos, "floor": math.floor, "ceil": math.ceil,
}
_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.Name, ast.Constant, ast.Load,
          ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
          ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
MAX_CHARS = 400


def check(src: str, variables: tuple[str, ...]) -> str | None:
    """Return None if `src` is an acceptable expression in `variables`, else the reason it is rejected."""
    if not isinstance(src, str) or not src.strip():
        return "empty"
    if len(src) > MAX_CHARS:
        return f"longer than {MAX_CHARS} characters"
    try:
        tree = ast.parse(src.strip(), mode="eval")
    except SyntaxError as e:
        return f"not a Python expression ({e.msg})"
    for node in ast.walk(tree):
        if not isinstance(node, _NODES):
            return f"{type(node).__name__} is not allowed"
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in FUNCS and not node.keywords):
            return "only abs, min, max, sqrt, exp, log, sin, cos, floor, ceil may be called"
        if isinstance(node, ast.Name) and node.id not in variables and node.id not in FUNCS:
            return f"unknown name {node.id!r}"
        if isinstance(node, ast.Constant) and not (isinstance(node.value, (int, float)) and not isinstance(node.value, complex) and abs(node.value) <= 1e9):
            return "only numbers up to 1e9 are allowed"
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow) and not (isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)) and abs(node.right.value) <= 4):
            return "** needs a constant exponent of at most 4"
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow) and any(isinstance(m, ast.BinOp) and isinstance(m.op, ast.Pow) for m in ast.walk(node.left)):
            return "nested ** is not allowed"
    return None


def evaluator(src: str, variables: tuple[str, ...]):
    """A function of the variables (positional) that evaluates the checked expression; raises ValueError if rejected."""
    why = check(src, variables)
    if why:
        raise ValueError(why)
    code = compile(ast.parse(src.strip(), mode="eval"), "<priority>", "eval")

    def f(*values):
        try:
            v = float(eval(code, {"__builtins__": {}}, {**FUNCS, **{k: float(v) for k, v in zip(variables, values)}}))  # noqa: S307 (whitelisted AST only)
        except (ArithmeticError, ValueError, OverflowError):
            return 0.0
        return v if math.isfinite(v) else 0.0

    return f
