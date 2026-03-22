import numpy as np

from src.envs.generators.base import Generator
from src.envs.generators.utils import compute_iterable_metrics


class PolynomialRootsGenerator(Generator):

    def __init__(self, params):
        self.min_degree = params.poly_min_degree
        self.max_degree = params.poly_max_degree
        self.complex_prob = params.poly_complex_prob
        self.root_range = params.poly_root_range

    @staticmethod
    def _sort_roots(roots):
        return np.array(sorted(roots, key=lambda z: (z.real, z.imag)))

    def generate(self, rng, is_train):
        d = int(rng.integers(self.min_degree, self.max_degree + 1))

        roots = []
        i = 0
        while i < d:
            if i + 1 < d and rng.random() < self.complex_prob:
                re = rng.uniform(-self.root_range, self.root_range)
                im = rng.uniform(0.5, self.root_range)
                roots.append(complex(re, im))
                roots.append(complex(re, -im))
                i += 2
            else:
                r = int(rng.integers(-self.root_range, self.root_range + 1))
                roots.append(complex(r, 0))
                i += 1

        roots = np.array(roots, dtype=np.complex128)

        coeffs = np.poly(roots)

        int_coeffs = np.round(coeffs.real).astype(np.int64)

        recovered = np.roots(int_coeffs.astype(np.float64))
        sorted_roots = self._sort_roots(roots)
        sorted_recovered = self._sort_roots(recovered)

        if len(sorted_recovered) != len(sorted_roots):
            return None

        if not np.allclose(sorted_recovered, sorted_roots, atol=1e-3, rtol=1e-3):
            return None

        return int_coeffs, None, sorted_roots

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if not isinstance(hyp, np.ndarray):
            return {"is_valid": 0}

        sorted_hyp = self._sort_roots(hyp)
        sorted_ans = self._sort_roots(answer)

        if len(sorted_hyp) != len(sorted_ans):
            return {"is_valid": 0}

        result = compute_iterable_metrics(sorted_hyp, sorted_ans, metrics, "correct", atol=1e-3, rtol=0.05)
        if "is_valid" not in result:
            result["is_valid"] = result["all_correct"]
        return result
