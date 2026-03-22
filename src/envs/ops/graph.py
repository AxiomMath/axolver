from src.envs.generators import FindShortestPathGenerator, LaplacianEigenvaluesGenerator, MaxCliqueGenerator
from src.envs.tokenizers import FloatTokenizer, GraphNodeListTokenizer, GraphTokenizer, NumberArrayTokenizer
from src.utils import bool_flag


def build_find_shortest_path(params):
    return {
        "problem_tokenizer": GraphTokenizer(params.max_nodes, params.weighted),
        "answer_tokenizer": GraphNodeListTokenizer(params.max_nodes),
        "query_tokenizer": GraphNodeListTokenizer(params.max_nodes),
        "generator": FindShortestPathGenerator(params),
    }


def build_laplacian_eigenvalues(params):
    float_tok = FloatTokenizer(params.base, params.float_precision, params.max_exponent)
    eigenvalue_tok = NumberArrayTokenizer(params.max_nodes, "V", 1, float_tok, encode_dim=True)
    return {
        "problem_tokenizer": GraphTokenizer(params.max_nodes, params.weighted),
        "answer_tokenizer": eigenvalue_tok,
        "generator": LaplacianEigenvaluesGenerator(params),
    }


def build_max_clique(params):
    return {
        "problem_tokenizer": GraphTokenizer(params.max_nodes, params.weighted),
        "answer_tokenizer": GraphNodeListTokenizer(params.max_nodes),
        "generator": MaxCliqueGenerator(params),
    }


def register_graph_args(parser):
    parser.add_argument("--max_nodes", type=int, default=8)
    parser.add_argument("--max_edges", type=int, default=15)
    parser.add_argument("--weighted", type=bool_flag, default=False)


def register_laplacian_args(parser):
    register_graph_args(parser)
    parser.add_argument("--float_precision", type=int, default=3)
    parser.add_argument("--max_exponent", type=int, default=10)
    parser.add_argument("--base", type=int, default=10)


OPERATIONS = {
    "find_shortest_path": {"build": build_find_shortest_path, "register_args": register_graph_args},
    "laplacian_eigenvalues": {"build": build_laplacian_eigenvalues, "register_args": register_laplacian_args},
    "max_clique": {"build": build_max_clique, "register_args": register_graph_args},
}
