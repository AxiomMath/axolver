import math

import numpy as np

from src.envs.tokenizers.base import Tokenizer
from src.envs.tokenizers.symbolic_int import SymbolicIntTokenizer


class FloatTokenizer(Tokenizer):
    """
    Float where the mantissa is an integer and the exponent is a symbolic integer.

    e.g. 3.14 with precision=3, base=10 -> ["FLOAT+", "3", "1", "4", "E-2"]
         -0.5 with precision=1, base=10 -> ["FLOAT-", "5", "E-1"]
    """

    def __init__(self, base, precision, max_exponent):
        self.base = base
        self.precision = precision
        self.max_exponent = max_exponent
        self.exponent_tokenizer = SymbolicIntTokenizer(-self.max_exponent, self.max_exponent, prefix="E")
        self._digits = [str(i) for i in range(self.base)]
        self._symbols = ["FLOAT+", "FLOAT-"] + self._digits + self.exponent_tokenizer.symbols

    @property
    def symbols(self):
        return list(self._symbols)

    def _decompose(self, value):
        """
        Decompose a float into (mantissa_int, exponent_int) such that
        value ≈ mantissa * 10^exponent, with mantissa having `precision` significant digits.
        """
        if value == 0.0:
            return 0, 0
        sign = -1 if value < 0 else 1
        abs_val = abs(value)
        exponent = math.floor(math.log10(abs_val)) - (self.precision - 1)
        mantissa = round(abs_val / (10**exponent))
        while mantissa != 0 and mantissa % 10 == 0:
            mantissa //= 10
            exponent += 1
        if exponent < -self.max_exponent or exponent > self.max_exponent:
            raise ValueError(f"Float value {value} has exponent {exponent} outside representable range [-{self.max_exponent}, {self.max_exponent}]")
        return sign * mantissa, exponent

    def encode(self, value):
        value = float(value)
        mantissa, exponent = self._decompose(value)
        tag = "FLOAT-" if mantissa < 0 else "FLOAT+"
        # Encode absolute mantissa digits
        m = abs(mantissa)
        if m == 0:
            digits = ["0"]
        else:
            digits = []
            while m > 0:
                digits.append(str(m % self.base))
                m //= self.base
            digits = digits[::-1]
        return [tag] + digits + self.exponent_tokenizer.encode(exponent)

    def parse(self, lst):
        if len(lst) == 0 or lst[0] not in ("FLOAT+", "FLOAT-"):
            return None, 0
        neg = lst[0] == "FLOAT-"
        # Parse mantissa digits
        mantissa = 0
        i = 0
        for x in lst[1:]:
            if not x.isdigit() or int(x) >= self.base:
                break
            mantissa = mantissa * self.base + int(x)
            i += 1
        if i == 0:
            return None, 0
        # Parse exponent
        exponent, e_pos = self.exponent_tokenizer.parse(lst[1 + i :])
        if exponent is None:
            return None, 0
        if neg:
            mantissa = -mantissa
        return mantissa * (10**exponent), 1 + i + e_pos


class FP15Tokenizer(Tokenizer):
    """
    Base-10 scientific-notation single-token float encoding.
    """

    def __init__(self, precision=2, max_exponent=16):
        self.float_precision = precision
        self.max_exponent = max_exponent
        assert (
            self.float_precision + self.max_exponent
        ) % 2 == 0, f"(precision + max_exponent) must be even, got {self.float_precision} + {self.max_exponent}"

        dig = 10**self.float_precision
        self.logrange = (self.float_precision + self.max_exponent) // 2
        self.base = 10 ** (self.logrange - self.float_precision)
        self.limit = 10**self.logrange

        syms = ["NaN", "-NaN"]
        syms.extend(["N" + str(i) + "e0" for i in range(-dig + 1, dig)])
        # exponent > 0
        for i in range(self.max_exponent):
            for j in range(1, 10):
                for k in range(dig):
                    syms.append("N" + str(j * dig + k) + "e" + str(i))
                    syms.append("N-" + str(j * dig + k) + "e" + str(i))
        self._symbols = syms

    def encode(self, value):
        value = float(value)
        if abs(value) > self.limit:
            return ["NaN"] if value > 0 else ["-NaN"]
        sign = -1 if value < 0 else 1
        v = abs(value) * self.base
        if v == 0:
            return ["N0e0"]
        e = int(math.log10(v))
        if e < 0:
            e = 0
        m = int(v * (10 ** (self.float_precision - e)) + 0.5)
        if m == 0:
            sign = 1
        if m == 10 ** (self.float_precision + 1):
            m = 10**self.float_precision
            e += 1
        if e >= self.max_exponent:
            return ["NaN"] if value > 0 else ["-NaN"]
        pref = "N" if sign == 1 else "N-"
        return [pref + str(m) + "e" + str(e)]

    def parse(self, lst):
        if len(lst) == 0:
            return None, 0
        tok = lst[0]
        if tok == "NaN":
            return float(self.limit), 1
        if tok == "-NaN":
            return float(-self.limit), 1
        if not tok.startswith("N"):
            return None, 0
        try:
            m_str, e_str = tok[1:].split("e")
            m = int(m_str)
            e = int(e_str)
            value = (m * (10**e)) / self.limit
            return value, 1
        except (ValueError, IndexError):
            return None, 0

    @property
    def symbols(self):
        return list(self._symbols)
