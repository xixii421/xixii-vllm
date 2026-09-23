import pytest
import torch

from sampler import Sampler, SamplingParams


def test_greedy_sampling_selects_argmax_from_each_batch_row() -> None:
    logits = torch.tensor([[0.2, 3.0, 1.0], [4.0, -2.0, 0.0]])

    token_ids = Sampler()(logits, SamplingParams(temperature=0))

    torch.testing.assert_close(token_ids, torch.tensor([1, 0]))


def test_three_dimensional_logits_use_only_last_sequence_position() -> None:
    logits = torch.tensor([[[100.0, 0.0], [0.0, 2.0]]])

    token_ids = Sampler()(logits, SamplingParams(temperature=0))

    torch.testing.assert_close(token_ids, torch.tensor([1]))


def test_seeded_sampling_matches_temperature_scaled_distribution() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0], [2.0, 0.5, -1.0]])
    params = SamplingParams(temperature=0.5)
    actual_generator = torch.Generator().manual_seed(17)
    expected_generator = torch.Generator().manual_seed(17)

    actual = Sampler()(logits, params, generator=actual_generator)
    expected_probabilities = torch.softmax(logits.float() / params.temperature, dim=-1)
    expected = torch.multinomial(
        expected_probabilities,
        num_samples=1,
        generator=expected_generator,
    ).squeeze(-1)

    torch.testing.assert_close(actual, expected)


def test_top_k_excludes_tokens_outside_k_largest_logits() -> None:
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]]).expand(256, -1)

    token_ids = Sampler()(
        logits,
        SamplingParams(top_k=2),
        generator=torch.Generator().manual_seed(3),
    )

    assert set(token_ids.tolist()) == {0, 1}


def test_top_p_keeps_minimal_prefix_reaching_probability_mass() -> None:
    probabilities = torch.tensor([[0.60, 0.25, 0.10, 0.05]])
    logits = probabilities.log().expand(256, -1)

    token_ids = Sampler()(
        logits,
        SamplingParams(top_p=0.70),
        generator=torch.Generator().manual_seed(5),
    )

    assert set(token_ids.tolist()) == {0, 1}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": float("inf")}, "temperature"),
        ({"top_k": -1}, "top_k"),
        ({"top_k": 1.5}, "top_k"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.1}, "top_p"),
    ],
)
def test_sampling_params_reject_invalid_values(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SamplingParams(**kwargs)


@pytest.mark.parametrize(
    "logits",
    [
        torch.ones(3),
        torch.ones(1, 2, 3, 4),
        torch.ones(0, 3),
        torch.ones(1, 0),
        torch.ones(1, 0, 3),
    ],
)
def test_sampler_rejects_invalid_shapes(logits: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        Sampler()(logits)


def test_sampler_rejects_non_floating_or_non_finite_logits() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        Sampler()(torch.tensor([[1, 2, 3]]))
    with pytest.raises(ValueError, match="finite"):
        Sampler()(torch.tensor([[0.0, float("nan")]]), SamplingParams(temperature=0))
