from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """
    Base class for tokenizers, encodes and decodes values to/from token sequences.
    Abstract methods for encoding/decoding, with a parse pattern for composability.
    """

    @abstractmethod
    def encode(self, val):
        pass

    @abstractmethod
    def parse(self, lst):
        """
        Parse a value from the beginning of a token list.
        Returns (value, num_consumed_tokens).
        Returns (None, 0) if parsing fails.
        """
        pass

    def decode(self, lst):
        v, p = self.parse(lst)
        if p == 0:
            return None
        return v

    @property
    @abstractmethod
    def symbols(self):
        pass
