import torch

from robobytes.model import VOCAB, ByteHNet, TokenLM, ema, suppress


def test_ema_matches_sequential():
    torch.manual_seed(0)
    z, P = torch.randn(3, 40, 8), torch.rand(3, 40) * 0.5 + 0.5
    P[:, 0] = 1
    seq, prev = [], torch.zeros(3, 8)
    for j in range(40):
        prev = P[:, j:j + 1] * z[:, j] + (1 - P[:, j:j + 1]) * prev
        seq.append(prev)
    assert torch.allclose(ema(z, P), torch.stack(seq, 1), atol=1e-5)


def test_suppress_refractory():
    b = torch.tensor([[1, 1, 0, 0, 1, 1, 0, 0, 0, 1]], dtype=torch.bool)
    assert suppress(b, 3).int().tolist() == [[1, 0, 0, 0, 1, 0, 0, 0, 0, 1]]


def test_models_causal_and_shapes():
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB, (2, 64))
    for m in (TokenLM(VOCAB, 64, 2, 4), ByteHNet(64, 2, 4, d_outer=32, outer_blocks=2)):
        m.eval()
        out = m(ids)["logits"]
        assert out.shape == (2, 64, VOCAB)
        ids2 = ids.clone()
        ids2[:, 40:] = (ids2[:, 40:] + 1) % VOCAB  # changing the future must not change outputs before it
        assert torch.allclose(m(ids2)["logits"][:, :40], out[:, :40], atol=1e-5)
