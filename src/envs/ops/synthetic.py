from src.envs.generators.synthetic import BracketMatchGenerator, CopyGenerator, DeduplicateGenerator, ParityGenerator, ReverseGenerator, SortGenerator
from src.envs.tokenizers import NumberArrayTokenizer, SymbolicIntTokenizer


def build_copy(params):
    tok = NumberArrayTokenizer(
        max_dim=params.copy_max_len, dim_prefix="V", tensor_dim=1, value_tokenizer=SymbolicIntTokenizer(0, params.copy_n_tokens - 1)
    )
    return {"problem_tokenizer": tok, "answer_tokenizer": tok, "generator": CopyGenerator(params)}


def build_reverse(params):
    tok = NumberArrayTokenizer(
        max_dim=params.reverse_max_len, dim_prefix="V", tensor_dim=1, value_tokenizer=SymbolicIntTokenizer(0, params.reverse_n_tokens - 1)
    )
    return {"problem_tokenizer": tok, "answer_tokenizer": tok, "generator": ReverseGenerator(params)}


def build_sort(params):
    tok = NumberArrayTokenizer(
        max_dim=params.sort_max_len, dim_prefix="V", tensor_dim=1, value_tokenizer=SymbolicIntTokenizer(0, params.sort_n_tokens - 1)
    )
    return {"problem_tokenizer": tok, "answer_tokenizer": tok, "generator": SortGenerator(params)}


def build_parity(params):
    in_tok = NumberArrayTokenizer(
        max_dim=params.parity_max_len, dim_prefix="V", tensor_dim=1, value_tokenizer=SymbolicIntTokenizer(0, params.parity_n_tokens - 1)
    )
    out_tok = NumberArrayTokenizer(max_dim=1, dim_prefix="V", tensor_dim=1, value_tokenizer=SymbolicIntTokenizer(0, 1))
    return {"problem_tokenizer": in_tok, "answer_tokenizer": out_tok, "generator": ParityGenerator(params)}


def build_deduplicate(params):
    tok = NumberArrayTokenizer(
        max_dim=params.deduplicate_max_len, dim_prefix="V", tensor_dim=1, value_tokenizer=SymbolicIntTokenizer(0, params.deduplicate_n_tokens - 1)
    )
    return {"problem_tokenizer": tok, "answer_tokenizer": tok, "generator": DeduplicateGenerator(params)}


def build_bracket_match(params):
    max_len = 2 * params.bracket_match_max_pairs
    in_tok = NumberArrayTokenizer(max_dim=max_len, dim_prefix="V", tensor_dim=1, value_tokenizer=SymbolicIntTokenizer(0, 1))
    out_tok = NumberArrayTokenizer(max_dim=max_len, dim_prefix="V", tensor_dim=1, value_tokenizer=SymbolicIntTokenizer(0, max_len - 1))
    return {"problem_tokenizer": in_tok, "answer_tokenizer": out_tok, "generator": BracketMatchGenerator(params)}


def register_copy_args(parser):
    parser.add_argument("--copy_max_len", type=int, default=20, help="Max sequence length for copy task")
    parser.add_argument("--copy_n_tokens", type=int, default=10, help="Number of distinct tokens (digits 0..n-1)")


def register_reverse_args(parser):
    parser.add_argument("--reverse_max_len", type=int, default=20, help="Max sequence length for reverse task")
    parser.add_argument("--reverse_n_tokens", type=int, default=10, help="Number of distinct tokens for reverse task")


def register_sort_args(parser):
    parser.add_argument("--sort_max_len", type=int, default=20, help="Max sequence length for sort task")
    parser.add_argument("--sort_n_tokens", type=int, default=10, help="Number of distinct tokens for sort task")


def register_parity_args(parser):
    parser.add_argument("--parity_max_len", type=int, default=20, help="Max sequence length for parity task")
    parser.add_argument("--parity_n_tokens", type=int, default=10, help="Number of distinct tokens for parity task")


def register_deduplicate_args(parser):
    parser.add_argument("--deduplicate_max_len", type=int, default=20, help="Max sequence length for deduplicate task")
    parser.add_argument("--deduplicate_n_tokens", type=int, default=10, help="Number of distinct tokens for deduplicate task")


def register_bracket_match_args(parser):
    parser.add_argument("--bracket_match_max_pairs", type=int, default=10, help="Max number of bracket pairs (sequence length = 2 * this)")


OPERATIONS = {
    "copy": {"build": build_copy, "register_args": register_copy_args},
    "reverse": {"build": build_reverse, "register_args": register_reverse_args},
    "sort": {"build": build_sort, "register_args": register_sort_args},
    "parity": {"build": build_parity, "register_args": register_parity_args},
    "deduplicate": {"build": build_deduplicate, "register_args": register_deduplicate_args},
    "bracket_match": {"build": build_bracket_match, "register_args": register_bracket_match_args},
}
