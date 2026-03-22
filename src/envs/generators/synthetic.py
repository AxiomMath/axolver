import numpy as np

from src.envs.generators.base import Generator


class CopyGenerator(Generator):
    def __init__(self, params):
        self.max_len = params.copy_max_len
        self.n_tokens = params.copy_n_tokens

    def generate(self, rng, is_train):
        length = rng.integers(1, self.max_len + 1)
        seq = rng.integers(0, self.n_tokens, length).tolist()
        return seq, None, seq

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if isinstance(hyp, np.ndarray):
            hyp = hyp.tolist()
        if isinstance(answer, np.ndarray):
            answer = answer.tolist()
        if hyp == answer:
            return {"is_valid": 1}
        return {"is_valid": 0}


class ReverseGenerator(Generator):
    def __init__(self, params):
        self.max_len = params.reverse_max_len
        self.n_tokens = params.reverse_n_tokens

    def generate(self, rng, is_train):
        length = rng.integers(1, self.max_len + 1)
        seq = rng.integers(0, self.n_tokens, length).tolist()
        return seq, None, seq[::-1]

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if isinstance(hyp, np.ndarray):
            hyp = hyp.tolist()
        if isinstance(answer, np.ndarray):
            answer = answer.tolist()
        if hyp == answer:
            return {"is_valid": 1}
        return {"is_valid": 0}


class SortGenerator(Generator):
    def __init__(self, params):
        self.max_len = params.sort_max_len
        self.n_tokens = params.sort_n_tokens

    def generate(self, rng, is_train):
        length = rng.integers(1, self.max_len + 1)
        seq = rng.integers(0, self.n_tokens, length).tolist()
        return seq, None, sorted(seq)

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if isinstance(hyp, np.ndarray):
            hyp = hyp.tolist()
        if isinstance(answer, np.ndarray):
            answer = answer.tolist()
        if hyp == answer:
            return {"is_valid": 1}
        return {"is_valid": 0}


class ParityGenerator(Generator):
    def __init__(self, params):
        self.max_len = params.parity_max_len
        self.n_tokens = params.parity_n_tokens

    def generate(self, rng, is_train):
        length = rng.integers(1, self.max_len + 1)
        seq = rng.integers(0, self.n_tokens, length).tolist()
        parity = [sum(seq) % 2]
        return seq, None, parity

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if isinstance(hyp, np.ndarray):
            hyp = hyp.tolist()
        if isinstance(answer, np.ndarray):
            answer = answer.tolist()
        if hyp == answer:
            return {"is_valid": 1}
        return {"is_valid": 0}


class DeduplicateGenerator(Generator):
    def __init__(self, params):
        self.max_len = params.deduplicate_max_len
        self.n_tokens = params.deduplicate_n_tokens

    def generate(self, rng, is_train):
        length = rng.integers(1, self.max_len + 1)
        seq = rng.integers(0, self.n_tokens, length).tolist()
        seen = set()
        deduped = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                deduped.append(x)
        return seq, None, deduped

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if isinstance(hyp, np.ndarray):
            hyp = hyp.tolist()
        if isinstance(answer, np.ndarray):
            answer = answer.tolist()
        if hyp == answer:
            return {"is_valid": 1}
        return {"is_valid": 0}


class BracketMatchGenerator(Generator):
    def __init__(self, params):
        self.max_pairs = params.bracket_match_max_pairs

    def generate(self, rng, is_train):
        n_pairs = rng.integers(1, self.max_pairs + 1)
        seq, match = self._generate_balanced(rng, n_pairs)
        return seq, None, match

    def _generate_balanced(self, rng, n_pairs):
        length = 2 * n_pairs
        seq = []
        match = [0] * length
        stack = []
        opens_left = n_pairs
        for i in range(length):
            can_open = opens_left > 0
            can_close = len(stack) > 0
            if can_open and can_close:
                do_open = rng.random() < 0.5
            elif can_open:
                do_open = True
            else:
                do_open = False
            if do_open:
                seq.append(0)
                stack.append(i)
                opens_left -= 1
            else:
                seq.append(1)
                j = stack.pop()
                match[i] = j
                match[j] = i
        return seq, match

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if isinstance(hyp, np.ndarray):
            hyp = hyp.tolist()
        if isinstance(answer, np.ndarray):
            answer = answer.tolist()
        if hyp == answer:
            return {"is_valid": 1}
        return {"is_valid": 0}
