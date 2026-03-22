from collections import OrderedDict

import sympy as sp
from sympy.calculus.util import AccumBounds
from sympy.parsing.sympy_parser import parse_expr

from src.envs.tokenizers.base import Tokenizer
from src.envs.tokenizers.integer import IntegerTokenizer


class InvalidPrefixExpression(Exception):
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return repr(self.data)


class UnknownSymPyOperator(Exception):
    pass


SYMPY_OPERATORS = {
    sp.Add: "add",
    sp.Mul: "mul",
    sp.Pow: "pow",
    sp.exp: "exp",
    sp.log: "ln",
    sp.Abs: "abs",
    sp.sign: "sign",
    # trig
    sp.sin: "sin",
    sp.cos: "cos",
    sp.tan: "tan",
    sp.cot: "cot",
    sp.sec: "sec",
    sp.csc: "csc",
    # trig inverses
    sp.asin: "asin",
    sp.acos: "acos",
    sp.atan: "atan",
    sp.acot: "acot",
    sp.asec: "asec",
    sp.acsc: "acsc",
    # hyperbolic
    sp.sinh: "sinh",
    sp.cosh: "cosh",
    sp.tanh: "tanh",
    sp.coth: "coth",
    sp.sech: "sech",
    sp.csch: "csch",
    # hyperbolic inverses
    sp.asinh: "asinh",
    sp.acosh: "acosh",
    sp.atanh: "atanh",
    sp.acoth: "acoth",
    sp.asech: "asech",
    sp.acsch: "acsch",
}

# operator name -> arity
OPERATORS = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "pow": 2,
    "rac": 2,
    "inv": 1,
    "pow2": 1,
    "pow3": 1,
    "pow4": 1,
    "pow5": 1,
    "sqrt": 1,
    "exp": 1,
    "ln": 1,
    "abs": 1,
    "sign": 1,
    # trig
    "sin": 1,
    "cos": 1,
    "tan": 1,
    "cot": 1,
    "sec": 1,
    "csc": 1,
    # trig inverses
    "asin": 1,
    "acos": 1,
    "atan": 1,
    "acot": 1,
    "asec": 1,
    "acsc": 1,
    # hyperbolic
    "sinh": 1,
    "cosh": 1,
    "tanh": 1,
    "coth": 1,
    "sech": 1,
    "csch": 1,
    # hyperbolic inverses
    "asinh": 1,
    "acosh": 1,
    "atanh": 1,
    "acoth": 1,
    "asech": 1,
    "acsch": 1,
}

_SPECIAL_UNARY = {"abs", "inv", "pow2", "pow3", "pow4", "pow5"}
_UNARY_FN_NAMES = {k for k, v in OPERATORS.items() if v == 1 and k not in _SPECIAL_UNARY}

# All known variable/coefficient/constant names for prefix parsing
KNOWN_SYMBOLS = {"x", "y", "z", "t", "pi", "E", "I"} | {f"a{i}" for i in range(10)}


def write_infix(token, args):
    if token == "add":
        return f"({args[0]})+({args[1]})"
    elif token == "sub":
        return f"({args[0]})-({args[1]})"
    elif token == "mul":
        return f"({args[0]})*({args[1]})"
    elif token == "div":
        return f"({args[0]})/({args[1]})"
    elif token == "pow":
        return f"({args[0]})**({args[1]})"
    elif token == "rac":
        return f"({args[0]})**(1/({args[1]}))"
    elif token == "abs":
        return f"Abs({args[0]})"
    elif token == "inv":
        return f"1/({args[0]})"
    elif token == "pow2":
        return f"({args[0]})**2"
    elif token == "pow3":
        return f"({args[0]})**3"
    elif token == "pow4":
        return f"({args[0]})**4"
    elif token == "pow5":
        return f"({args[0]})**5"
    elif token in _UNARY_FN_NAMES:
        return f"{token}({args[0]})"
    raise InvalidPrefixExpression(f"Unknown token in prefix expression: {token}, with arguments {args}")


def _prefix_to_infix(expr, int_tok):
    if len(expr) == 0:
        raise InvalidPrefixExpression("Empty prefix list.")
    t = expr[0]
    if t in OPERATORS:
        args = []
        rest = expr[1:]
        for _ in range(OPERATORS[t]):
            sub, rest = _prefix_to_infix(rest, int_tok)
            args.append(sub)
        return write_infix(t, args), rest
    elif t in KNOWN_SYMBOLS or t.startswith("a"):
        return t, expr[1:]
    else:
        val, consumed = int_tok.parse(expr)
        if val is None:
            raise InvalidPrefixExpression(f"Cannot parse prefix expression at: {expr[:5]}")
        return str(val), expr[consumed:]


def prefix_to_infix(expr, int_tok):
    p, r = _prefix_to_infix(expr, int_tok)
    if len(r) > 0:
        raise InvalidPrefixExpression(f'Incorrect prefix expression. "{r}" was not parsed.')
    return f"({p})"


def _sympy_to_prefix(op, expr, int_tok):
    n_args = len(expr.args)
    assert (op in ("add", "mul") and n_args >= 2) or (op not in ("add", "mul") and 1 <= n_args <= 2)
    # sqrt shorthand
    if op == "pow" and isinstance(expr.args[1], sp.Rational) and expr.args[1].p == 1 and expr.args[1].q == 2:
        return ["sqrt"] + sympy_to_prefix(expr.args[0], int_tok)
    parse_list = []
    for i in range(n_args):
        if i == 0 or i < n_args - 1:
            parse_list.append(op)
        parse_list += sympy_to_prefix(expr.args[i], int_tok)
    return parse_list


def sympy_to_prefix(expr, int_tok):
    if isinstance(expr, sp.Symbol):
        return [str(expr)]
    elif isinstance(expr, sp.Integer):
        return int_tok.encode(int(str(expr)))
    elif isinstance(expr, sp.Rational):
        return ["div"] + int_tok.encode(int(expr.p)) + int_tok.encode(int(expr.q))
    elif expr == sp.E:
        return ["E"]
    elif expr == sp.pi:
        return ["pi"]
    elif expr == sp.I:
        return ["I"]
    for op_type, op_name in SYMPY_OPERATORS.items():
        if isinstance(expr, op_type):
            return _sympy_to_prefix(op_name, expr, int_tok)
    raise UnknownSymPyOperator(f"Unknown SymPy operator: {expr}")


class SymbolicSequenceTokenizer(Tokenizer):
    def __init__(self, int_base, max_int):
        self.int_base = int_base
        self.max_int = max_int
        self.int_tokenizer = IntegerTokenizer(int_base)

        self.constants = ["pi", "E", "I"]
        self.variables = OrderedDict(
            {
                "x": sp.Symbol("x", real=True, nonzero=True),
                "y": sp.Symbol("y", real=True, nonzero=True),
                "z": sp.Symbol("z", real=True, nonzero=True),
                "t": sp.Symbol("t", real=True, nonzero=True),
            }
        )
        self.coefficients = OrderedDict({f"a{i}": sp.Symbol(f"a{i}", real=True) for i in range(10)})

        self.local_dict = {}
        for k, v in list(self.variables.items()) + list(self.coefficients.items()):
            self.local_dict[k] = v

        self._symbols = (
            list(OPERATORS.keys()) + list(self.variables.keys()) + list(self.coefficients.keys()) + self.constants + self.int_tokenizer.symbols
        )

    @property
    def symbols(self):
        return list(self._symbols)

    def encode(self, expr):
        return sympy_to_prefix(expr, self.int_tokenizer)

    def parse(self, lst):
        try:
            infix_str, remaining = _prefix_to_infix(lst, self.int_tokenizer)
            consumed = len(lst) - len(remaining)
            infix_str = f"({infix_str})"
            expr = parse_expr(infix_str, evaluate=True, local_dict=self.local_dict)
            if expr.has(AccumBounds):
                return None, 0
            return expr, consumed
        except (InvalidPrefixExpression, Exception):
            return None, 0
