"""CPU-only correctness checks with synthetic tokens; no training data."""
import pytest
import torch
from models.gpt import GPT, GPTConfig


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_shapes_and_causality(backend):
    torch.manual_seed(7)
    model = GPT(GPTConfig(19, 8, 2, 2, 16, dropout=0, attention_backend=backend)).eval()
    tokens = torch.randint(19, (2, 8))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, 8, 19)
    assert loss.ndim == 0 and torch.isfinite(loss)
    changed = tokens.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 19
    updated, _ = model(changed)
    torch.testing.assert_close(logits[:, :4], updated[:, :4])
    with pytest.raises(ValueError, match="block_size"):
        model(torch.zeros(1, 9, dtype=torch.long))


def test_manual_sdpa_forward_and_gradient_parity():
    torch.manual_seed(3)
    a = GPT(GPTConfig(19, 8, 1, 2, 16, dropout=0, attention_backend="manual"))
    b = GPT(GPTConfig(19, 8, 1, 2, 16, dropout=0, attention_backend="sdpa"))
    b.load_state_dict(a.state_dict())
    tokens = torch.randint(19, (2, 6))
    x, la = a(tokens, tokens)
    y, lb = b(tokens, tokens)
    torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(la, lb)
    la.backward(); lb.backward()
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa.grad, pb.grad, atol=1e-6, rtol=1e-4)
