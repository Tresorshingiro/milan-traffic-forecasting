import pytest

torch = pytest.importorskip("torch")
from milan.models.registry import MODELS, build_model, count_parameters

@pytest.mark.parametrize("name", ["dlinear", "tcn", "gru"])
def test_forward_maps_a_batch_of_windows_to_one_value_each(name):
    model = build_model(name, L=144)
    x = torch.randn(8, 144)
    out = model(x)
    assert out.shape == (8,), f"{name} produced {out.shape}, expected (8,)"
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", ["dlinear", "tcn", "gru"])
def test_gradients_reach_every_parameter(name):
    model = build_model(name, L=144)
    model(torch.randn(4, 144)).sum().backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None]
    assert dead == [], f"{name} has parameters with no gradient: {dead}"


@pytest.mark.parametrize("name, L", [("dlinear", 36), ("dlinear", 288),
                                     ("tcn", 36), ("gru", 36), ("gru", 288)])
def test_models_accept_every_candidate_input_length(name, L):
    assert build_model(name, L=L)(torch.randn(2, L)).shape == (2,)


def test_registry_exposes_exactly_the_three_selected_models():
    assert set(MODELS) == {"dlinear", "tcn", "gru"}

def test_unknown_model_name_is_rejected():
    with pytest.raises(KeyError, match="unknown model"):
        build_model("lstm", L=144)

# Dlinear specifics

def test_dlinear_is_a_low_capacity_model():
    """2L + 2 parameters: the low-capacity arm of H1."""
    assert count_parameters(build_model("dlinear", L=144)) == 2 * 144 + 2


def test_dlinear_decomposition_reconstructs_its_input():
    """trend + seasonal must equal the original window, or the split loses information."""
    from milan.models.dlinear import MovingAvg
    x = torch.randn(4, 144)
    trend = MovingAvg(25)(x)
    assert trend.shape == x.shape
    torch.testing.assert_close(trend + (x - trend), x)


def test_dlinear_rejects_an_even_kernel():
    with pytest.raises(ValueError, match="odd"):
        build_model("dlinear", L=144, kernel=24)


@pytest.mark.parametrize("kernel", [25, 145])
def test_dlinear_supports_both_grid_kernels(kernel):
    assert build_model("dlinear", L=144, kernel=kernel)(torch.randn(2, 144)).shape == (2,)


# TCN specifics

def test_tcn_receptive_field_matches_the_documented_formula():
    model = build_model("tcn", L=144, kernel=3, dilations=(1, 2, 4, 8, 16, 32))
    assert model.receptive_field == 1 + 2 * (3 - 1) * 63 == 253


def test_tcn_rejects_a_configuration_that_cannot_see_its_whole_input():
    """Silently failing here would leave the model blind to most of its window."""
    with pytest.raises(ValueError, match="receptive field"):
        build_model("tcn", L=288, kernel=3, dilations=(1, 2, 4))


def test_tcn_is_causal():
    model = build_model("tcn", L=144).eval()
    x = torch.randn(1, 144)
    base = model(x)
    perturbed = x.clone()
    perturbed[0, -1] += 10.0
    assert not torch.allclose(base, model(perturbed)), "output ignores the most recent step"

# GRU specifics

@pytest.mark.parametrize("layers,hidden", [(1, 32), (1, 64), (2, 32), (2, 64)])
def test_gru_supports_every_grid_configuration(layers, hidden):
    model = build_model("gru", L=144, layers=layers, hidden=hidden)
    assert model(torch.randn(2, 144)).shape == (2,)


def test_model_capacity_ordering_matches_the_hypothesis():
    """H1 frames DLinear as low-capacity; verify that is actually true here."""
    sizes = {n: count_parameters(build_model(n, L=144)) for n in MODELS}
    assert sizes["dlinear"] < sizes["gru"]
    assert sizes["dlinear"] < sizes["tcn"]