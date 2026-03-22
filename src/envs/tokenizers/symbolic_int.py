from src.envs.tokenizers.base import Tokenizer


class SymbolicIntTokenizer(Tokenizer):
    """
    One token per integer from min_val to max_val.
    Optionally prefixed, e.g. "E-100"..."E100" for exponents, "V1"..."V10" for dimensions.
    """

    def __init__(self, min_val, max_val, prefix=""):
        self.prefix = prefix
        self.min_val = min_val
        self.max_val = max_val
        self._symbols = [self.prefix + str(i) for i in range(min_val, max_val + 1)]

    @property
    def symbols(self):
        return list(self._symbols)

    def encode(self, value):
        return [self.prefix + str(int(value))]

    def parse(self, lst):
        if len(lst) == 0 or lst[0] not in self._symbols:
            return None, 0
        return int(lst[0][len(self.prefix) :]), 1
