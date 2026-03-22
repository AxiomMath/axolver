import numpy as np

from src.envs.tokenizers.base import Tokenizer
from src.envs.tokenizers.complex import ComplexTokenizer
from src.envs.tokenizers.float import FloatTokenizer, FP15Tokenizer
from src.envs.tokenizers.symbolic_int import SymbolicIntTokenizer


def _infer_dtype(value_tokenizer):
    if isinstance(value_tokenizer, ComplexTokenizer):
        return np.complex128
    elif isinstance(value_tokenizer, (FloatTokenizer, FP15Tokenizer)):
        return np.float64
    return np.int64


class NumberArrayTokenizer(Tokenizer):
    """
    Array of values of any shape. Encodes shape dimensions first, then all elements.
    The value_tokenizer determines how individual elements are tokenized.
    """

    def __init__(self, max_dim, dim_prefix, tensor_dim, value_tokenizer, encode_dim=True):
        self.tensor_dim = tensor_dim
        self.encode_dim = encode_dim
        self.dim_tokenizer = SymbolicIntTokenizer(1, max_dim, dim_prefix)
        self.value_tokenizer = value_tokenizer
        self.dtype = _infer_dtype(value_tokenizer)
        if encode_dim:
            self._symbols = self.dim_tokenizer.symbols + self.value_tokenizer.symbols
        else:
            self._symbols = self.value_tokenizer.symbols

    @property
    def symbols(self):
        return list(self._symbols)

    def encode(self, vector):
        lst = []
        assert len(np.shape(vector)) == self.tensor_dim
        if self.encode_dim:
            for d in np.shape(vector):
                lst.extend(self.dim_tokenizer.encode(d))
        for val in np.nditer(np.array(vector)):
            lst.extend(self.value_tokenizer.encode(val))
        return lst

    def parse(self, lst):
        # Two distinct parsing strategies: with encoded dims we read shape first, without we greedily consume values.
        if self.encode_dim:
            shap = []
            pos = 0
            for _ in range(self.tensor_dim):
                v, p = self.dim_tokenizer.parse(lst[pos:])
                if v is None:
                    return None, 0
                shap.append(v)
                pos += p
            m = np.zeros(tuple(shap), dtype=self.dtype)
            for val in np.nditer(m, op_flags=["readwrite"]):
                v, p = self.value_tokenizer.parse(lst[pos:])
                if v is None:
                    return None, 0
                pos += p
                val[...] = v
            return m, pos
        else:
            values = []
            pos = 0
            while pos < len(lst):
                v, p = self.value_tokenizer.parse(lst[pos:])
                if p == 0:
                    break
                values.append(v)
                pos += p
            if len(values) == 0:
                return None, 0
            return np.array(values, dtype=self.dtype), pos
