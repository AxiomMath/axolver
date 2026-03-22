from logging import getLogger

from src.model.rnn import RNNSeq2Seq
from src.model.transformer import TransformerSeq2Seq

logger = getLogger()


def check_model_params(params):
    assert params.model_type in ("transformer", "lstm", "gru"), f"Unknown model_type: {params.model_type}"
    assert params.architecture in ("encoder_decoder", "encoder_only", "decoder_only"), f"Unknown architecture: {params.architecture}"

    if params.model_type in ("lstm", "gru"):
        assert (
            params.architecture == "encoder_decoder"
        ), f"model_type={params.model_type} only supports encoder_decoder architecture, got {params.architecture}"
    else:
        # Transformer-specific validations
        assert params.enc_emb_dim % params.n_enc_heads == 0
        assert params.dec_emb_dim % params.n_dec_heads == 0
        assert params.norm in ("layernorm", "rmsnorm", "rmsnorm_no_params"), f"Unknown norm: {params.norm}"
        assert params.activation in ("relu", "relu_squared", "gelu"), f"Unknown activation: {params.activation}"
        assert params.enc_pos_emb in ("abs_sinusoidal", "abs_learned", "none"), f"Unknown enc_pos_emb: {params.enc_pos_emb}"
        assert params.dec_pos_emb in ("abs_sinusoidal", "abs_learned", "none"), f"Unknown dec_pos_emb: {params.dec_pos_emb}"


def build_model(params):
    if params.model_type == "transformer":
        model = TransformerSeq2Seq(params)
    else:
        model = RNNSeq2Seq(params)

    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.to(params.device)
    return model
