import math

import numpy as np

from src.envs.generators.base import Generator
from src.envs.generators.utils import integer_loguniform_sequence, integer_sequence


class GCDGenerator(Generator):

    def __init__(self, params):
        self.minint = params.minint
        self.maxint = params.maxint
        self.operand_distribution = params.operand_distribution
        self.outcome_distribution = params.outcome_distribution
        self.max_gcd = params.max_gcd
        self.mixed_pct = params.mixed_pct
        assert self.maxint >= self.max_gcd
        assert self.operand_distribution in ["uniform", "log_uniform"]
        assert self.outcome_distribution in ["uniform", "log_uniform", "natural", "inv_sqrt", "mixed"]
        if self.outcome_distribution == "inv_sqrt":
            # Precompute inverse-sqrt distribution
            dist = np.array([1.0 / math.sqrt(i + 1) for i in range(self.max_gcd)])
            self._inv_sqrt_dist = dist / dist.sum()

    def _sample_operands(self, rng, is_train, maxint):
        if is_train and self.operand_distribution == "log_uniform":
            return integer_loguniform_sequence(self.minint, maxint, 2, rng)
        else:
            return integer_sequence(self.minint, maxint, 2, rng)

    def _sample_with_target_gcd(self, target_gcd, rng, is_train):
        while True:
            inp = self._sample_operands(rng, is_train, maxint=self.maxint // target_gcd)
            if math.gcd(int(inp[0]), int(inp[1])) == 1:
                break
        return np.array([target_gcd * int(inp[0]), target_gcd * int(inp[1])])

    def generate(self, rng, is_train):
        if not is_train:
            out = int(rng.integers(1, self.max_gcd + 1))
            inp = self._sample_with_target_gcd(out, rng, is_train)
        elif self.outcome_distribution == "natural":
            inp = self._sample_operands(rng, is_train, self.maxint)
            out = math.gcd(int(inp[0]), int(inp[1]))
        elif self.outcome_distribution == "uniform":
            out = int(rng.integers(1, self.max_gcd + 1))
            inp = self._sample_with_target_gcd(out, rng, is_train)
        elif self.outcome_distribution == "log_uniform":
            lgs = math.log10(self.max_gcd) * rng.random()
            out = max(1, int(10**lgs))
            inp = self._sample_with_target_gcd(out, rng, is_train)
        elif self.outcome_distribution == "inv_sqrt":
            out = int(rng.choice(range(1, self.max_gcd + 1), p=self._inv_sqrt_dist))
            inp = self._sample_with_target_gcd(out, rng, is_train)
        elif self.outcome_distribution == "mixed":
            if rng.random() < self.mixed_pct:
                # uniform mixed_pct of the time
                out = int(rng.integers(1, self.max_gcd + 1))
                inp = self._sample_with_target_gcd(out, rng, is_train)
            else:
                # natural (1 - mixed_pct) of the time
                inp = self._sample_operands(rng, is_train, self.maxint)
                out = math.gcd(int(inp[0]), int(inp[1]))
        return inp, None, out

    def encode_class_id(self, problem_data, question_data, answer_data):
        return answer_data

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if hyp == answer:
            return {"is_valid": 1}
        return {"is_valid": 0}


class FractionSimplifyGenerator(Generator):

    def __init__(self, params):
        self.minint = params.minint
        self.maxint = params.maxint

    def generate(self, rng, is_train):
        integers = integer_sequence(self.minint, self.maxint, 3, rng)
        k = integers[0]
        if k == 1:
            k = rng.integers(2, self.maxint + 1)
        a, b = integers[1], integers[2]
        g = math.gcd(a, b)
        inp = [k * a // g, k * b // g]
        out = [a // g, b // g]
        return inp, None, out

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if np.array_equal(hyp, answer):
            return {"is_valid": 1}
        return {"is_valid": 0}


class FractionRoundGenerator(Generator):

    def __init__(self, params):
        self.minint = params.minint
        self.maxint = params.maxint

    def generate(self, rng, is_train):
        integers = integer_sequence(self.minint, self.maxint, 3, rng)
        m1 = min(integers[1], integers[2])
        m2 = max(integers[1], integers[2])
        if m2 == m1:
            m1 = m2 - 1
        inp = [integers[0] * m2 + m1, m2]
        out = integers[0]
        return inp, None, out

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if hyp == answer:
            return {"is_valid": 1}
        return {"is_valid": 0}


class FractionAddGenerator(Generator):

    def __init__(self, params):
        self.minint = params.minint
        self.maxint = params.maxint

    def generate(self, rng, is_train):
        inp = integer_sequence(self.minint, self.maxint, 4, rng)
        num = inp[0] * inp[3] + inp[1] * inp[2]
        den = inp[1] * inp[3]
        g = math.gcd(num, den)
        out = [int(num // g), int(den // g)]
        return inp, None, out

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if np.array_equal(hyp, answer):
            return {"is_valid": 1}
        return {"is_valid": 0}


class FractionProductGenerator(Generator):

    def __init__(self, params):
        self.minint = params.minint
        self.maxint = params.maxint

    def generate(self, rng, is_train):
        inp = integer_sequence(self.minint, self.maxint, 4, rng)
        num = inp[0] * inp[2]
        den = inp[1] * inp[3]
        g = math.gcd(num, den)
        out = [int(num // g), int(den // g)]
        return inp, None, out

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if np.array_equal(hyp, answer):
            return {"is_valid": 1}
        return {"is_valid": 0}


class FractionDeterminantGenerator(Generator):

    def __init__(self, params):
        self.minint = params.minint
        self.maxint = params.maxint

    def generate(self, rng, is_train):
        inp = integer_sequence(self.minint, self.maxint, 4, rng)
        out = inp[0] * inp[3] - inp[1] * inp[2]
        return inp, None, out

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if hyp == answer:
            return {"is_valid": 1}
        return {"is_valid": 0}


class FractionCompareGenerator(Generator):

    def __init__(self, params):
        self.minint = params.minint
        self.maxint = params.maxint

    def generate(self, rng, is_train):
        inp = integer_sequence(self.minint, self.maxint, 4, rng)
        cmp = inp[0] * inp[3] - inp[1] * inp[2]
        out = 1 if cmp > 0 else 0
        return inp, None, out

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if hyp == answer:
            return {"is_valid": 1}
        return {"is_valid": 0}


class ModularAddGenerator(Generator):

    def __init__(self, params):
        self.modulus = params.modulus
        self.minint = params.minint
        self.maxint = params.maxint

    def generate(self, rng, is_train):
        inp = integer_sequence(self.minint, self.maxint, 2, rng)
        out = (inp[0] + inp[1]) % self.modulus
        return inp, None, out

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if hyp == answer:
            return {"is_valid": 1}
        return {"is_valid": 0}


class ModularMulGenerator(Generator):

    def __init__(self, params):
        self.modulus = params.modulus
        self.minint = params.minint
        self.maxint = params.maxint

    def generate(self, rng, is_train):
        inp = integer_sequence(self.minint, self.maxint, 2, rng)
        out = (inp[0] * inp[1]) % self.modulus
        return inp, None, out

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if hyp == answer:
            return {"is_valid": 1}
        return {"is_valid": 0}
