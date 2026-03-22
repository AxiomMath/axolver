from src.envs.tokenizers.base import Tokenizer


class GraphTokenizer(Tokenizer):
    def __init__(self, max_nodes, weighted, weight_tokenizer=None):
        self.max_nodes = max_nodes
        self.weighted = weighted
        self.weight_tokenizer = weight_tokenizer
        self._node_symbols = [f"N{i}" for i in range(self.max_nodes)]
        self._graph_symbols = [f"G{i}" for i in range(self.max_nodes + 1)]
        self._symbols = self._graph_symbols + self._node_symbols
        if self.weighted:
            self._symbols = self._symbols + self.weight_tokenizer.symbols

    @property
    def symbols(self):
        return list(self._symbols)

    def encode(self, graph):
        num_nodes, edges = graph
        tokens = [f"G{num_nodes}"]
        for edge in edges:
            tokens.append(f"N{edge[0]}")
            tokens.append(f"N{edge[1]}")
            if self.weighted and len(edge) > 2:
                tokens.extend(self.weight_tokenizer.encode(edge[2]))
        return tokens

    def _parse_node(self, token):
        if token in self._node_symbols:
            return int(token[1:])
        return None

    def parse(self, lst):
        if len(lst) == 0 or lst[0] not in self._graph_symbols:
            return None, 0
        num_nodes = int(lst[0][1:])
        edges = []
        pos = 1
        while pos < len(lst):
            if pos + 1 >= len(lst):
                break
            ni = self._parse_node(lst[pos])
            if ni is None:
                break
            nj = self._parse_node(lst[pos + 1])
            if nj is None:
                break
            pos += 2
            if self.weighted:
                w, w_pos = self.weight_tokenizer.parse(lst[pos:])
                if w is None:
                    break
                edges.append((ni, nj, w))
                pos += w_pos
            else:
                edges.append((ni, nj))
        return (num_nodes, edges), pos


class GraphNodeListTokenizer(Tokenizer):
    def __init__(self, max_nodes):
        self.max_nodes = max_nodes
        self._node_symbols = [f"N{i}" for i in range(self.max_nodes)]

    @property
    def symbols(self):
        return list(self._node_symbols)

    def encode(self, nodes):
        return [f"N{node}" for node in nodes]

    def parse(self, lst):
        nodes = []
        pos = 0
        while pos < len(lst):
            if lst[pos] not in self._node_symbols:
                break
            nodes.append(int(lst[pos][1:]))
            pos += 1
        if len(nodes) == 0:
            return None, 0
        return nodes, pos
