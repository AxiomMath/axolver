from collections import deque

import numpy as np

from src.envs.generators.base import Generator
from src.envs.generators.utils import (
    _bfs_path,
    _build_adjacency,
    _is_valid_path,
    _laplacian_eigenvalues,
    _max_clique,
    compute_iterable_metrics,
    random_connected_graph,
)


class FindShortestPathGenerator(Generator):

    def __init__(self, params):
        self.max_nodes = params.max_nodes
        self.max_edges = params.max_edges

    def generate(self, rng, is_train):
        n, edges = random_connected_graph(self.max_nodes, self.max_edges, rng)
        adj = _build_adjacency(n, edges)
        start, end = rng.integers(0, n, 2)
        while start == end and n > 1:
            end = rng.integers(0, n)
        path = _bfs_path(adj, start, end)
        if path is None:
            return None
        return (n, edges), (start, end), path

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        n, edges = problem
        start, end = question
        is_valid_p = _is_valid_path(edges, start, end, hyp)
        if not is_valid_p:
            return {"is_valid": 0}
        adj = _build_adjacency(n, edges)
        shortest = _bfs_path(adj, start, end)
        is_correct = shortest is not None and len(hyp) == len(shortest)

        metrics_dict = {"is_valid": 1 if is_correct else 0}
        if "path_length_ratio" in metrics and shortest is not None:
            metrics_dict["path_length_ratio"] = float(len(shortest)) / len(hyp)
        return metrics_dict


class LaplacianEigenvaluesGenerator(Generator):

    def __init__(self, params):
        self.max_nodes = params.max_nodes
        self.max_edges = params.max_edges

    def generate(self, rng, is_train):
        n, edges = random_connected_graph(self.max_nodes, self.max_edges, rng)
        eigenvalues = _laplacian_eigenvalues(n, edges)
        return (n, edges), None, eigenvalues

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if not isinstance(hyp, np.ndarray) or len(hyp) != len(answer):
            return {"is_valid": 0}
        result = compute_iterable_metrics(hyp, answer, metrics, "correct", atol=1e-4, sort=True)
        if "is_valid" not in result:
            result["is_valid"] = result["all_correct"]
        return result


class MaxCliqueGenerator(Generator):

    def __init__(self, params):
        self.max_nodes = params.max_nodes
        self.max_edges = params.max_edges

    def generate(self, rng, is_train):
        n, edges = random_connected_graph(self.max_nodes, self.max_edges, rng)
        clique = _max_clique(n, edges, rng)
        return (n, edges), None, clique

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        if not isinstance(hyp, list):
            return {"is_valid": 0}
        n, edges = problem
        edge_set = set()
        for u, v in edges:
            edge_set.add((u, v))
            edge_set.add((v, u))
        is_clique = True
        for i in range(len(hyp)):
            for j in range(i + 1, len(hyp)):
                if (hyp[i], hyp[j]) not in edge_set:
                    is_clique = False
                    break
            if not is_clique:
                break

        if not is_clique:
            return {"is_valid": 0}
        is_correct = len(hyp) == len(answer)

        metrics_dict = {"is_valid": 1 if is_correct else 0}
        if "size_ratio" in metrics:
            metrics_dict["size_ratio"] = float(len(hyp)) / len(answer)
        return metrics_dict
