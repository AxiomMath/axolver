import math
from collections import deque

import numpy as np

# metrics when answer is an iterable


def compute_iterable_metrics(hyp, answer, metrics, metric, atol=1e-4, rtol=0.0, sort=False):
    """
    Compute the following metrics for ordered sequences:
    - how many elements are equal (both number and ratio)
    - if all elements are equal
    - which elements are equal
    If sort=True, both hyp and answer are sorted before comparison.
    """
    if sort:
        hyp = np.sort(hyp)
        answer = np.sort(answer)
    close = np.isclose(hyp, answer, atol=atol, rtol=rtol)
    nb = int(close.sum())
    metrics_dict = {}
    if f"all_{metric}" in metrics:
        metrics_dict[f"all_{metric}"] = int(close.all())
    else:
        metrics_dict["is_valid"] = int(close.all())
    if f"ith_{metric}" in metrics:
        for i, c in enumerate(close):
            metrics_dict[f"{metric}_{i}"] = int(c)
    if f"ratio_{metric}" in metrics:
        metrics_dict[f"ratio_{metric}"] = nb / len(answer)
    if f"nb_{metric}" in metrics:
        metrics_dict[f"nb_{metric}"] = nb
    return metrics_dict


# integers generator utils


def integer_sequence(minint, maxint, length, rng):
    """Uniform random integers from minint to maxint (inclusive)."""
    return rng.integers(minint, maxint + 1, length)


def integer_loguniform_sequence(minint, maxint, length, rng):
    """Log-uniform random integers from minint to maxint."""
    lgs = math.log10(minint) + (math.log10(maxint) - math.log10(minint)) * rng.random(length)
    return np.maximum(np.int64(10**lgs), minint)


def integer_matrix(maxint, n, p, rng):
    """Random integer (n, p) matrix with coefficients in [-maxint, maxint]."""
    maxint = int(maxint + 0.5)
    return rng.integers(-maxint, maxint + 1, (n, p))


# graph generator utils


def _build_adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _is_connected(n, edges):
    if n <= 1:
        return True
    adj = _build_adjacency(n, edges)
    visited = {0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return len(visited) == n


def _spanning_tree_graph(n, max_edges, rng):
    nodes = list(range(n))
    rng.shuffle(nodes)
    edges = set()
    for i in range(1, n):
        u, v = nodes[i - 1], nodes[i]
        edges.add((min(u, v), max(u, v)))
    max_extra = min(max_edges - len(edges), n * (n - 1) // 2 - len(edges))
    if max_extra > 0:
        n_extra = rng.integers(0, max_extra + 1)
        for _ in range(n_extra * 3):
            u, v = rng.integers(0, n, 2)
            if u != v:
                e = (min(u, v), max(u, v))
                if e not in edges:
                    edges.add(e)
                    if len(edges) - (n - 1) >= n_extra:
                        break
    return list(edges)


def _erdos_renyi_graph(n, max_edges, rng, max_attempts=20):
    max_possible = n * (n - 1) // 2
    min_edges = n - 1
    target = rng.integers(min_edges, min(max_edges, max_possible) + 1)
    p = target / max_possible if max_possible > 0 else 1.0

    for _ in range(max_attempts):
        edges = set()
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    edges.add((i, j))
        edge_list = list(edges)
        if len(edge_list) <= max_edges and _is_connected(n, edge_list):
            return edge_list
    return None


def random_connected_graph(max_nodes, max_edges, rng):
    """
    Sample a random connected undirected graph.
    Randomly chooses between spanning-tree method and Erdos-Renyi with rejection.
    """
    n = rng.integers(2, max_nodes + 1)
    if rng.integers(2) == 0:
        edges = _spanning_tree_graph(n, max_edges, rng)
    else:
        edges = _erdos_renyi_graph(n, max_edges, rng)
        if edges is None:
            edges = _spanning_tree_graph(n, max_edges, rng)
    return n, edges


def _bfs_path(adj, start, end):
    visited = {start}
    queue = deque([(start, [start])])
    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))
    return None


def _is_valid_path(edges, start, end, path):
    if not path or path[0] != start or path[-1] != end:
        return False
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
    for i in range(len(path) - 1):
        if (path[i], path[i + 1]) not in edge_set:
            return False
    return True


def _laplacian_eigenvalues(n, edges):
    L = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        L[u, v] -= 1
        L[v, u] -= 1
        L[u, u] += 1
        L[v, v] += 1
    eigenvalues = np.linalg.eigvalsh(L)
    eigenvalues[eigenvalues < 1e-10] = 0.0
    return np.sort(eigenvalues)


def _max_clique(n, edges, rng):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    cliques = []

    def bron_kerbosch(R, P, X):
        if not P and not X:
            cliques.append(sorted(R))
            return
        pivot = max(P | X, key=lambda v: len(adj[v] & P))
        for v in list(P - adj[pivot]):
            bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)

    bron_kerbosch(set(), set(range(n)), set())

    max_size = max(len(c) for c in cliques)
    max_cliques = [c for c in cliques if len(c) == max_size]
    return list(max_cliques[rng.integers(len(max_cliques))])
