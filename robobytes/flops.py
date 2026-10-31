"""Analytic training FLOPs per byte (brief section 4): 6 x parameters used per position, plus the attention term
6 x layers x context x width (Kaplan et al.'s 2 n_layer n_ctx d per forward position, x3 for training), counting only
the positions each network actually sees. Embedding lookups and learned positions cost 0."""
from .model import ByteHNet, TokenLM


def _p(module):
    return sum(p.numel() for n, p in module.named_parameters() if not n.startswith("pos") and ".pos." not in n)


def main_cfg(m):
    blk = m.main.blocks
    return len(blk), blk[0].qkv.in_features


def flops_per_byte(model, ctx_bytes=1024, compression=1.0):
    """compression = bytes per main-network position (1 for Byte, R_BPE for BPE, realised N for chunked models)."""
    L, d = main_cfg(model)
    ctx = ctx_bytes / compression
    attn = 6 * L * ctx * d
    if isinstance(model, TokenLM):
        per_pos = 6 * (_p(model.main) + _p(model.head)) + attn
        return per_pos / compression
    assert isinstance(model, ByteHNet)
    outer = sum(_p(x) for x in (model.encoder, model.decoder, model.router_norm, model.wq, model.wk, model.residual,
                                model.head))
    main = _p(model.main) + _p(model.up_proj) + _p(model.down_proj)
    dechunk = 6 * ctx * model.down_proj.out_features  # EMA as a K x K mix of width-d_outer chunks, per chunk
    return 6 * outer + (6 * main + attn + dechunk) / compression


def params(model):
    return sum(p.numel() for p in model.parameters())
