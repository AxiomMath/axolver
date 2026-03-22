import numpy as np

from src.envs.generators.base import Generator
from src.envs.generators.utils import compute_iterable_metrics, integer_matrix, integer_sequence


class MatrixTransposeGenerator(Generator):

    def __init__(self, params):
        self.dim1 = params.dim1
        self.dim2 = params.dim2
        self.maxint = params.maxint

    def generate(self, rng, is_train):
        n = rng.integers(2, self.dim1 + 1)
        p = rng.integers(2, self.dim2 + 1)
        m = integer_matrix(self.maxint, n, p, rng)
        return m, None, m.T.copy()

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if not isinstance(hyp, np.ndarray) or hyp.shape != answer.shape:
            return {"is_valid": 0}
        if np.array_equal(hyp, answer):
            return {"is_valid": 1}
        return {"is_valid": 0}


class MatrixSumGenerator(Generator):

    def __init__(self, params):
        self.dim1 = params.dim1
        self.dim2 = params.dim2
        self.maxint = params.maxint

    def generate(self, rng, is_train):
        n = rng.integers(2, self.dim1 + 1)
        p = rng.integers(2, self.dim2 + 1)
        a = integer_matrix(self.maxint, n, p, rng)
        b = integer_matrix(self.maxint, n, p, rng)
        problem = np.vstack([a, b])
        return problem, None, (a + b)

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if not isinstance(hyp, np.ndarray) or hyp.shape != answer.shape:
            return {"is_valid": 0}
        if np.array_equal(hyp, answer):
            return {"is_valid": 1}
        return {"is_valid": 0}


class MatrixVectorGenerator(Generator):

    def __init__(self, params):
        self.dim1 = params.dim1
        self.dim2 = params.dim2
        self.maxint = params.maxint

    def generate(self, rng, is_train):
        n = rng.integers(2, self.dim1 + 1)
        p = rng.integers(2, self.dim2 + 1)
        a = integer_matrix(self.maxint, n, p, rng)
        v = integer_sequence(-self.maxint, self.maxint, p, rng)
        problem = np.vstack([a, v.reshape(1, -1)])
        result = a @ v
        return problem, None, result

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if not isinstance(hyp, np.ndarray) or hyp.shape != answer.shape:
            return {"is_valid": 0}
        if np.array_equal(hyp, answer):
            return {"is_valid": 1}
        return {"is_valid": 0}


class MatrixDeterminantGenerator(Generator):

    def __init__(self, params):
        self.dim1 = params.dim1
        self.maxint = params.maxint

    def generate(self, rng, is_train):
        n = rng.integers(2, self.dim1 + 1)
        m = integer_matrix(self.maxint, n, n, rng)
        det = int(round(float(np.linalg.det(m))))
        return m, None, det

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        try:
            if int(hyp) == int(answer):
                return {"is_valid": 1}
            return {"is_valid": 0}
        except (TypeError, ValueError):
            return {"is_valid": 0}


class MatrixEigenvaluesGenerator(Generator):

    def __init__(self, params):
        self.dim1 = params.dim1
        self.maxint = params.maxint
        self.rtol = params.rtol

    def generate(self, rng, is_train):
        n = rng.integers(2, self.dim1 + 1)
        a = integer_matrix(self.maxint, n, n, rng)
        m = np.tril(a) + np.tril(a, -1).T
        eigenvalues = np.sort(np.linalg.eigvalsh(m.astype(np.float64)))
        return m, None, eigenvalues

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if not isinstance(hyp, np.ndarray) or len(hyp) != len(answer):
            return {"is_valid": 0}
        hyp_s = np.sort(hyp)
        ans_s = np.sort(answer)
        result = compute_iterable_metrics(hyp_s, ans_s, metrics, "correct", atol=1e-4, rtol=self.rtol, sort=False)
        if "is_valid" not in result:
            result["is_valid"] = result["all_correct"]
        if "rel_l1" in metrics:
            diff = np.abs(hyp_s - ans_s)
            norm_tgt = np.max(np.abs(ans_s))
            result["rel_l1"] = float(np.max(diff) / (norm_tgt + 1e-12))
        return result


class MatrixInverseGenerator(Generator):

    def __init__(self, params):
        self.dim1 = params.dim1
        self.maxint = params.maxint
        self.max_exponent = params.max_exponent

    def generate(self, rng, is_train):
        n = rng.integers(2, self.dim1 + 1)
        a = integer_matrix(self.maxint, n, n, rng)
        m = np.tril(a) + np.tril(a, -1).T
        det = np.linalg.det(m.astype(np.float64))
        if abs(det) < 0.1:
            return None
        inv = np.linalg.inv(m.astype(np.float64))
        if np.any(np.abs(inv) > 10**self.max_exponent):
            return None
        return m, None, inv

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if not isinstance(hyp, np.ndarray) or hyp.shape != answer.shape:
            return {"is_valid": 0}
        n = problem.shape[0]
        m = problem.astype(np.float64)
        try:
            product = m @ hyp
            is_correct = np.allclose(product, np.eye(n), atol=0.01)
            metrics_dict = {"is_valid": 1 if is_correct else 0}
            if "frobenius_error" in metrics:
                metrics_dict["frobenius_error"] = float(np.linalg.norm(product - np.eye(n), "fro")) / n
            if "rel_l1" in metrics:
                residual = product - np.eye(n)
                metrics_dict["rel_l1"] = float(np.max(np.abs(residual)))
            return metrics_dict
        except (ValueError, np.linalg.LinAlgError):
            pass
        return {"is_valid": 0}


class MatrixRankGenerator(Generator):

    def __init__(self, params):
        self.dim1 = params.dim1
        self.dim2 = params.dim2
        self.maxint = params.maxint

    def generate(self, rng, is_train):
        maxrank = min(self.dim1, self.dim2)
        rank = rng.integers(1, maxrank + 1)
        P = integer_matrix(self.maxint, self.dim1, rank, rng)
        Q = integer_matrix(self.maxint, rank, self.dim2, rng)
        mat = P @ Q
        check_rank = np.linalg.matrix_rank(mat)
        if check_rank != rank:
            return None
        return mat, None, rank

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if hyp == answer:
            return {"is_valid": 1}
        return {"is_valid": 0}
