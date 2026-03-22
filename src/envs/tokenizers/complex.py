from src.envs.tokenizers.base import Tokenizer
from src.envs.tokenizers.float import FloatTokenizer


class ComplexTokenizer(Tokenizer):
    """
    Complex number tokenizer. Encodes as Re + Im parts using FloatTokenizer.
    e.g. (3.14 + 2.0j) -> ["Re", "FLOAT+", "3", "1", "4", "E-2", "Im", "FLOAT+", "2", "E0"]
    """

    def __init__(self, base, precision, max_exponent):
        self.float_tokenizer = FloatTokenizer(base, precision, max_exponent)
        self._symbols = ["Re", "Im"] + self.float_tokenizer.symbols

    @property
    def symbols(self):
        return list(self._symbols)

    def encode(self, value):
        value = complex(value)
        return ["Re"] + self.float_tokenizer.encode(value.real) + ["Im"] + self.float_tokenizer.encode(value.imag)

    def parse(self, lst):
        if len(lst) == 0 or lst[0] != "Re":
            return None, 0
        real, r_pos = self.float_tokenizer.parse(lst[1:])
        if real is None:
            return None, 0
        if 1 + r_pos >= len(lst) or lst[1 + r_pos] != "Im":
            return None, 0
        imag, i_pos = self.float_tokenizer.parse(lst[2 + r_pos :])
        if imag is None:
            return None, 0
        return complex(real, imag), 2 + r_pos + i_pos
