"""Offline regression tests for top-p and min-p filtering in entropy decoding."""

import torch

from optillm.entropy_decoding import _sample


def sampled_tokens(probs, num_samples=2000, **kwargs):
    logits = probs.log().view(1, 1, -1)
    generator = torch.Generator().manual_seed(0)
    return {
        _sample(logits, generator=generator, **kwargs).item()
        for _ in range(num_samples)
    }


def test_top_p_keeps_the_most_likely_token():
    # The 39 less likely tokens together exceed top_p.
    probs = torch.tensor([0.05] + [0.95 / 39] * 39)

    tokens = sampled_tokens(probs, temperature=1.0, top_p=0.9, top_k=40)

    assert 0 in tokens
    # The nucleus keeps tokens until those before them exceed 0.9: 36 of 40.
    assert tokens == set(range(36))


def test_min_p_removes_unlikely_tokens():
    probs = torch.tensor([0.5, 0.3] + [0.2 / 20] * 20)

    tokens = sampled_tokens(probs, temperature=1.0, top_p=1.0, top_k=22, min_p=0.1)

    assert tokens == {0, 1}
