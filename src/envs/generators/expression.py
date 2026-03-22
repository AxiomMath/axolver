import math
import re
from collections import OrderedDict
from logging import getLogger

import numpy as np
import sympy as sp
from sympy.calculus.util import AccumBounds
from sympy.parsing.sympy_parser import parse_expr

from src.envs.generators.base import Generator
from src.envs.tokenizers.symbolic_sequence import OPERATORS, prefix_to_infix, sympy_to_prefix
from src.utils import TimeoutError, timeout

logger = getLogger()


EVAL_SYMBOLS = {"x", "y", "z", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"}
EVAL_VALUES = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 2.1, 3.1]
EVAL_VALUES = EVAL_VALUES + [-x for x in EVAL_VALUES]


def _simplify(f, seconds=1):
    assert seconds > 0

    @timeout(seconds)
    def _inner(f):
        try:
            f2 = sp.simplify(f)
            if any(s.is_Dummy for s in f2.free_symbols):
                logger.warning(f"Detected Dummy symbol when simplifying {f} to {f2}")
                return f
            return f2
        except TimeoutError:
            return f
        except Exception as e:
            logger.warning(f"{type(e).__name__} exception when simplifying {f}")
            return f

    return _inner(f)


def reduce_coefficients(expr, variables, coefficients):
    temp = sp.Symbol("temp")
    while True:
        last = expr
        for a in coefficients:
            if a not in expr.free_symbols:
                continue
            for subexp in sp.preorder_traversal(expr):
                if a in subexp.free_symbols and not any(var in subexp.free_symbols for var in variables):
                    p = expr.subs(subexp, temp)
                    if a in p.free_symbols:
                        continue
                    else:
                        expr = p.subs(temp, a)
                        break
        if last == expr:
            break
    return expr


def reindex_coefficients(expr, coefficients):
    coeffs = sorted([x for x in expr.free_symbols if x in coefficients], key=lambda x: x.name)
    for idx, coeff in enumerate(coefficients):
        if idx >= len(coeffs):
            break
        if coeff != coeffs[idx]:
            expr = expr.subs(coeffs[idx], coeff)
    return expr


def simplify_const_with_coeff(expr, coeff):
    assert coeff.is_Atom
    for parent in sp.preorder_traversal(expr):
        if any(coeff == arg for arg in parent.args):
            break
    if not (parent.is_Add or parent.is_Mul):
        return expr
    removed = [arg for arg in parent.args if len(arg.free_symbols) == 0]
    if len(removed) > 0:
        removed = parent.func(*removed)
        new_coeff = (coeff - removed) if parent.is_Add else (coeff / removed)
        expr = expr.subs(coeff, new_coeff)
    return expr


EXP_OPERATORS = {"exp", "sinh", "cosh"}


def count_nested_exp(s):
    stack = []
    count = 0
    max_count = 0
    for v in re.findall("[+-/*//()]|[a-zA-Z0-9]+", s):
        if v == "(":
            stack.append(v)
        elif v == ")":
            while True:
                x = stack.pop()
                if x in EXP_OPERATORS:
                    count -= 1
                if x == "(":
                    break
        else:
            stack.append(v)
            if v in EXP_OPERATORS:
                count += 1
                max_count = max(max_count, count)
    assert len(stack) == 0
    return max_count


def is_valid_expr(s):
    s = s.replace("(E)", "(exp(1))")
    s = s.replace("(I)", "(1)")
    s = s.replace("(pi)", "(1)")
    s = re.sub(
        r"(?<![a-z])(Abs|sign|ln|sin|cos|tan|sec|csc|cot|asin|acos|atan|asec|acsc|acot|tanh|sech|csch|coth|asinh|acosh|atanh|asech|acoth|acsch)\(",
        "(",
        s,
    )
    if count_nested_exp(s) >= 4:
        return False
    for v in EVAL_VALUES:
        try:
            local_dict = {sym: (v + 1e-4 * i) for i, sym in enumerate(EVAL_SYMBOLS)}
            value = eval(
                s,
                {"__builtins__": {}},
                {
                    **local_dict,
                    "exp": math.exp,
                    "sqrt": math.sqrt,
                    "ln": math.log,
                    "log": math.log,
                    "pi": math.pi,
                    "E": math.e,
                    "abs": abs,
                    "Abs": abs,
                    "sign": lambda x: (1 if x > 0 else (-1 if x < 0 else 0)),
                },
            )
            if not (math.isnan(value) or math.isinf(value)):
                return True
        except (FloatingPointError, ZeroDivisionError, TypeError, MemoryError, OverflowError, ValueError):
            continue
    return False


class ExpressionGenerator(Generator):

    VARIABLES = OrderedDict({"x": sp.Symbol("x", real=True, nonzero=True)})

    COEFFICIENTS = OrderedDict({f"a{i}": sp.Symbol(f"a{i}", real=True) for i in range(10)})

    CONSTANTS = ["pi", "E"]

    def __init__(self, params, int_tokenizer):
        self.int_tokenizer = int_tokenizer
        self.max_int = params.max_int
        self.max_ops = params.max_ops
        self.max_len = params.max_len
        self.positive = params.positive
        self.n_variables = params.n_variables
        self.n_coefficients = params.n_coefficients

        ops = sorted([x.split(":") for x in params.operators.split(",")])
        for o, _ in ops:
            assert o in OPERATORS, f"Unknown operator: {o}"
        self.all_ops = [o for o, _ in ops]
        self.una_ops = [o for o, _ in ops if OPERATORS[o] == 1]
        self.bin_ops = [o for o, _ in ops if OPERATORS[o] == 2]
        self.all_ops_probs = np.array([float(w) for _, w in ops], dtype=np.float64)
        self.una_ops_probs = np.array([float(w) for o, w in ops if OPERATORS[o] == 1], dtype=np.float64)
        self.bin_ops_probs = np.array([float(w) for o, w in ops if OPERATORS[o] == 2], dtype=np.float64)
        self.all_ops_probs /= self.all_ops_probs.sum()
        self.una_ops_probs /= self.una_ops_probs.sum()
        self.bin_ops_probs /= self.bin_ops_probs.sum()

        self.leaf_probs = np.array([float(x) for x in params.leaf_probs.split(",")], dtype=np.float64)
        self.leaf_probs /= self.leaf_probs.sum()

        self.rewrite_functions = [x for x in params.rewrite_functions.split(",") if x != ""]

        self.nl = 1
        self.p1 = 1
        self.p2 = 1

        self.ubi_dist = self._generate_ubi_dist(self.max_ops)

        self.local_dict = {}
        for k, v in list(self.VARIABLES.items()) + list(self.COEFFICIENTS.items()):
            self.local_dict[k] = v

    def _generate_ubi_dist(self, max_ops):
        D = []
        D.append([0] + [self.nl**i for i in range(1, 2 * max_ops + 1)])
        for n in range(1, 2 * max_ops + 1):
            s = [0]
            for e in range(1, 2 * max_ops - n + 1):
                s.append(self.nl * s[e - 1] + self.p1 * D[n - 1][e] + self.p2 * D[n - 1][e + 1])
            D.append(s)
        assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
        D = [[D[j][i] for j in range(len(D)) if i < len(D[j])] for i in range(max(len(x) for x in D))]
        return D

    def _sample_next_pos_ubi(self, nb_empty, nb_ops, rng):
        assert nb_empty > 0 and nb_ops > 0
        probs = []
        for i in range(nb_empty):
            probs.append((self.nl**i) * self.p1 * self.ubi_dist[nb_empty - i][nb_ops - 1])
        for i in range(nb_empty):
            probs.append((self.nl**i) * self.p2 * self.ubi_dist[nb_empty - i + 1][nb_ops - 1])
        probs = [p / self.ubi_dist[nb_empty][nb_ops] for p in probs]
        probs = np.array(probs, dtype=np.float64)
        e = rng.choice(2 * nb_empty, p=probs)
        arity = 1 if e < nb_empty else 2
        e = e % nb_empty
        return e, arity

    def _get_leaf(self, rng):
        leaf_type = rng.choice(4, p=self.leaf_probs)
        if leaf_type == 0:
            return [list(self.VARIABLES.keys())[rng.integers(self.n_variables)]]
        elif leaf_type == 1:
            return [list(self.COEFFICIENTS.keys())[rng.integers(self.n_coefficients)]]
        elif leaf_type == 2:
            c = rng.integers(1, self.max_int + 1)
            c = c if (self.positive or rng.integers(2) == 0) else -c
            return self.int_tokenizer.encode(c)
        else:
            return [self.CONSTANTS[rng.integers(len(self.CONSTANTS))]]

    def _generate_expr(self, nb_total_ops, rng):
        stack = [None]
        nb_empty = 1
        l_leaves = 0
        t_leaves = 1

        for nb_ops in range(nb_total_ops, 0, -1):
            skipped, arity = self._sample_next_pos_ubi(nb_empty, nb_ops, rng)
            if arity == 1:
                op = rng.choice(self.una_ops, p=self.una_ops_probs)
            else:
                op = rng.choice(self.bin_ops, p=self.bin_ops_probs)
            nb_empty += OPERATORS[op] - 1 - skipped
            t_leaves += OPERATORS[op] - 1
            l_leaves += skipped
            pos = [i for i, v in enumerate(stack) if v is None][l_leaves]
            stack = stack[:pos] + [op] + [None for _ in range(OPERATORS[op])] + stack[pos + 1 :]

        leaves = [self._get_leaf(rng) for _ in range(t_leaves)]
        if not any(len(leaf) == 1 and leaf[0] == "x" for leaf in leaves):
            leaves[-1] = ["x"]
        rng.shuffle(leaves)

        for pos in range(len(stack) - 1, -1, -1):
            if stack[pos] is None:
                stack = stack[:pos] + leaves.pop() + stack[pos + 1 :]
        assert len(leaves) == 0
        return stack

    def _prefix_to_sympy(self, prefix):
        infix = prefix_to_infix(prefix, self.int_tokenizer)
        if not is_valid_expr(infix):
            return None
        expr = parse_expr(infix, evaluate=True, local_dict=self.local_dict)
        if expr.has(sp.I) or expr.has(AccumBounds):
            return None
        for f in self.rewrite_functions:
            if f == "expand":
                expr = sp.expand(expr)
            elif f == "factor":
                expr = sp.factor(expr)
            elif f == "expand_log":
                expr = sp.expand_log(expr, force=True)
            elif f == "logcombine":
                expr = sp.logcombine(expr, force=True)
            elif f == "powsimp":
                expr = sp.powsimp(expr, force=True)
            elif f == "simplify":
                expr = _simplify(expr, seconds=1)
        return expr

    def _sympy_to_prefix(self, expr):
        return sympy_to_prefix(expr, self.int_tokenizer)

    def _reduce_and_reindex(self, expr):
        x = self.VARIABLES["x"]
        coeffs = list(self.COEFFICIENTS.values())[: self.n_coefficients]
        if self.n_coefficients > 0:
            expr = reduce_coefficients(expr, [x], coeffs)
            for c in coeffs:
                expr = simplify_const_with_coeff(expr, c)
            expr = reindex_coefficients(expr, coeffs)
        return expr
