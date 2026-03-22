from src.envs.tokenizers.base import Tokenizer


class IntegerTokenizer(Tokenizer):
    """
    Single integers in base N (positive base), with INT+/INT- prefix.

    e.g. 42 in base 10 -> ["INT+", "4", "2"], -7 -> ["INT-", "7"]
    """

    def __init__(self, base):
        self.base = base
        assert self.base > 0

        self._digits = [str(i) for i in range(self.base)]
        self._sign_tokens = ["INT+", "INT-"]
        self._symbols = list(self._sign_tokens) + self._digits

    @property
    def symbols(self):
        return list(self._symbols)

    def encode(self, value):
        value = int(value)
        base = self.base
        neg = value < 0
        value = -value if neg else value
        res = []
        while True:
            rem = value % base
            value = value // base
            res.append(str(rem))
            if value == 0:
                break
        tag = "INT-" if neg else "INT+"
        return [tag] + res[::-1]

    def parse(self, lst):
        if len(lst) == 0 or lst[0] not in self._sign_tokens:
            return None, 0
        head = lst[0]
        base = self.base
        val = 0
        i = 0
        for x in lst[1:]:
            if not x.isdigit() or int(x) >= base:
                break
            val = val * base + int(x)
            i += 1
        if i == 0:
            return None, 0
        if head == "INT-":
            val = -val
        return val, i + 1
