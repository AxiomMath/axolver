import warnings
from logging import getLogger

import numpy as np
import sympy as sp
from sympy.integrals.risch import NonElementaryIntegral

from src.envs.generators.expression import EVAL_VALUES, ExpressionGenerator, _simplify
from src.envs.tokenizers.symbolic_sequence import OPERATORS
from src.utils import TimeoutError, timeout

logger = getLogger()


INTEGRAL_FUNC = {
    sp.erf,
    sp.erfc,
    sp.erfi,
    sp.erfinv,
    sp.erfcinv,
    sp.expint,
    sp.Ei,
    sp.li,
    sp.Li,
    sp.Si,
    sp.Ci,
    sp.Shi,
    sp.Chi,
    sp.fresnelc,
    sp.fresnels,
}


def has_inf_nan(*args):
    for f in args:
        if f.has(sp.nan) or f.has(sp.oo) or f.has(-sp.oo) or f.has(sp.zoo):
            return True
    return False


def remove_root_constant_terms(expr, variables, mode):
    variables = variables if isinstance(variables, list) else [variables]
    assert mode in ["add", "mul"]
    if not any(x in variables for x in expr.free_symbols):
        return expr
    if mode == "add" and expr.is_Add or mode == "mul" and expr.is_Mul:
        args = [arg for arg in expr.args if any(x in variables for x in arg.free_symbols)]
        if len(args) == 1:
            return args[0]
        elif len(args) < len(expr.args):
            return expr.func(*args)
    return expr


class IntegrationGenerator(ExpressionGenerator):

    def __init__(self, params, int_tokenizer):
        super().__init__(params, int_tokenizer)
        self.fwd_prob = params.fwd_prob

    def _gen_prim_fwd(self, rng):
        x = self.VARIABLES["x"]
        nb_ops = rng.integers(0, 3) if rng.integers(40) == 0 else rng.integers(3, self.max_ops + 1)

        try:

            @timeout(5)
            def _build():
                f_expr = self._generate_expr(nb_ops, rng)
                f = self._prefix_to_sympy(f_expr)
                if f is None or x not in f.free_symbols:
                    return None

                if rng.integers(2) == 0:
                    f = remove_root_constant_terms(f, x, "add")
                f = self._reduce_and_reindex(f)

                F = sp.integrate(f, x, risch=True)
                if isinstance(F, NonElementaryIntegral):
                    return None
                F = F.doit()
                if has_inf_nan(f, F) or isinstance(F, NonElementaryIntegral) or F.has(sp.Integral) or F.has(sp.Piecewise):
                    return None
                if any(op.func in INTEGRAL_FUNC for op in sp.preorder_traversal(F)):
                    return None

                f_prefix = self._sympy_to_prefix(f)
                F_prefix = self._sympy_to_prefix(F)
                if max(len(f_prefix), len(F_prefix)) + 2 > self.max_len:
                    return None

                real_nb_ops = sum(1 for tok in f_prefix if tok in OPERATORS)
                if real_nb_ops < nb_ops / 2:
                    return None

                return f, F

            result = _build()
            if result is None:
                return None
            f, F = result

        except (TimeoutError, ValueError, AttributeError, TypeError, OverflowError, NotImplementedError):
            return None
        except Exception as e:
            logger.debug(f"gen_prim_fwd exception: {type(e).__name__}: {e}")
            return None

        return f, None, F

    def _gen_prim_bwd(self, rng):
        x = self.VARIABLES["x"]
        nb_ops = rng.integers(0, 4) if rng.integers(40) == 0 else rng.integers(4, self.max_ops + 1)

        try:

            @timeout(5)
            def _build():
                F_expr = self._generate_expr(nb_ops, rng)
                F = self._prefix_to_sympy(F_expr)
                if F is None or x not in F.free_symbols:
                    return None

                F = remove_root_constant_terms(F, x, "add")
                F = self._reduce_and_reindex(F)

                f = sp.diff(F, x)
                if rng.integers(2) == 1:
                    f = _simplify(f, seconds=2)

                if has_inf_nan(f, F):
                    return None

                f_prefix = self._sympy_to_prefix(f)
                F_prefix = self._sympy_to_prefix(F)
                if max(len(f_prefix), len(F_prefix)) + 2 > self.max_len:
                    return None

                real_nb_ops = sum(1 for tok in F_prefix if tok in OPERATORS)
                if real_nb_ops < nb_ops / 2:
                    return None

                return f, F

            result = _build()
            if result is None:
                return None
            f, F = result

        except (TimeoutError, ValueError, AttributeError, TypeError, OverflowError, NotImplementedError):
            return None
        except Exception as e:
            logger.debug(f"gen_prim_bwd exception: {type(e).__name__}: {e}")
            return None

        return f, None, F

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}

        x = self.VARIABLES["x"]

        try:
            diff = _simplify(answer - hyp, seconds=2)
            if diff == 0:
                return {"is_valid": 1}

            hyp_deriv = sp.diff(hyp, x)
            check = _simplify(hyp_deriv - problem, seconds=2)
            if check == 0:
                return {"is_valid": 1}

            x_arr = np.array(EVAL_VALUES, dtype=np.float64)
            f_diff = sp.lambdify(x, diff, modules=["numpy"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with np.errstate(all="ignore"):
                    diff_arr = np.asarray(f_diff(x_arr), dtype=np.float64).ravel()
            mask = np.isfinite(diff_arr)
            if mask.sum() >= 2:
                vals = diff_arr[mask]
                if vals.max() - vals.min() < 1e-8:
                    return {"is_valid": 1}

        except (TimeoutError, Exception):
            pass

        return {"is_valid": 0}

    def generate(self, rng, is_train):
        if rng.random() < self.fwd_prob:
            return self._gen_prim_fwd(rng)
        else:
            return self._gen_prim_bwd(rng)
