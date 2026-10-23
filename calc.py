"""Safe calculator used as a tool by the chatbot (no eval, whitelisted operations only)."""

import ast
import math
import operator

_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "log": math.log, "ln": math.log, "log10": math.log10, "log2": math.log2,
    "exp": math.exp, "abs": abs, "round": round, "floor": math.floor,
    "ceil": math.ceil, "factorial": math.factorial, "radians": math.radians,
    "degrees": math.degrees, "pow": math.pow, "min": min, "max": max,
}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
MAX_EXP = 1000
MAX_LEN = 500


def _ev(n):
    if isinstance(n, ast.Expression):
        return _ev(n.body)
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
        return n.value
    if isinstance(n, ast.Name) and n.id in _CONSTS:
        return _CONSTS[n.id]
    if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
        return _UN[type(n.op)](_ev(n.operand))
    if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
        left, right = _ev(n.left), _ev(n.right)
        if isinstance(n.op, ast.Pow):
            if abs(right) > MAX_EXP:
                raise ValueError("exponent too large")
            both_int = isinstance(left, int) and isinstance(right, int)
            if not (both_int and abs(left) <= 10**6):
                left = float(left)
        return _BIN[type(n.op)](left, right)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
        args = [_ev(a) for a in n.args]
        if n.func.id == "factorial" and (len(args) != 1 or not isinstance(args[0], int) or args[0] > 1000):
            raise ValueError("factorial needs an integer <= 1000")
        return _FUNCS[n.func.id](*args)
    raise ValueError("unsupported expression")


def _fmt(x):
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x) if len(str(x)) <= 60 else f"{float(x):.12e}"
    if isinstance(x, float):
        if x.is_integer() and abs(x) < 1e15:
            return str(int(x))
        return f"{x:.12g}"
    return str(x)


def calculate(expression: str) -> str:
    """Evaluate a math expression and return 'expr = result' or 'Error: ...'."""
    try:
        expr = (expression or "").strip().replace("^", "**")
        if not expr:
            return "Error: empty expression"
        if len(expr) > MAX_LEN:
            return "Error: expression too long"
        result = _ev(ast.parse(expr, mode="eval"))
        return f"{expression.strip()} = {_fmt(result)}"
    except ZeroDivisionError:
        return "Error: division by zero"
    except OverflowError:
        return "Error: result too large"
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"
