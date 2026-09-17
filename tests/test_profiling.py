import numpy as np
import pytest

from milan.analysis.profiling import (
    acf_at, acf_ratio, burstiness, coefficient_of_variation, dfa_exponent,
    fano_factor, gini, permutation_entropy, profile_area, seasonal_strength,
    spectral_entropy,
)

RNG = np.random.default_rng(0)
NOISE = RNG.standard_normal(8928)
WALK = np.cumsum(RNG.standard_normal(8928))
TONE = np.sin(np.arange(8928) * 2 * np.pi / 144)


# --- permutation entropy: 0 for a monotone ramp, ~1 for white noise -------

def test_permutation_entropy_is_zero_for_a_monotone_ramp():
    assert permutation_entropy(np.arange(5000.0)) == pytest.approx(0.0, abs=1e-9)


def test_permutation_entropy_is_near_one_for_white_noise():
    assert permutation_entropy(NOISE) > 0.99


def test_permutation_entropy_is_low_for_a_pure_tone():
    """A deterministic periodic signal has few distinct ordinal patterns."""
    assert permutation_entropy(TONE) < 0.4


def test_permutation_entropy_is_bounded():
    for signal in (NOISE, TONE, WALK):
        assert 0.0 <= permutation_entropy(signal) <= 1.0


# --- spectral entropy: ~1 for white noise, 0 for a pure tone --------------

def test_spectral_entropy_is_near_one_for_white_noise():
    assert spectral_entropy(NOISE) > 0.9


def test_spectral_entropy_is_zero_for_a_pure_tone():
    assert spectral_entropy(TONE) == pytest.approx(0.0, abs=1e-6)


# --- DFA: alpha ~ 0.5 for noise, ~1.5 for an integrated signal ------------

def test_dfa_exponent_is_half_for_white_noise():
    assert dfa_exponent(NOISE) == pytest.approx(0.5, abs=0.08)


def test_dfa_exponent_is_three_halves_for_a_random_walk():
    """DFA integrates its input, so an already-integrated signal gives alpha = H + 1."""
    assert dfa_exponent(WALK) == pytest.approx(1.5, abs=0.15)


# --- burstiness, dispersion, concentration --------------------------------

def test_burstiness_is_minus_one_for_a_constant_series():
    assert burstiness(np.full(1000, 5.0)) == pytest.approx(-1.0)


def test_burstiness_is_near_zero_for_an_exponential_series():
    """sigma == mu for the exponential distribution, so B -> 0."""
    assert burstiness(RNG.exponential(1.0, 200_000)) == pytest.approx(0.0, abs=0.02)


def test_burstiness_is_bounded():
    assert -1.0 <= burstiness(np.abs(NOISE)) <= 1.0


def test_coefficient_of_variation_and_fano_hand_computed():
    x = np.tile([1.0, 2.0, 3.0, 4.0], 25)      # 100 points, mean 2.5
    assert coefficient_of_variation(x) == pytest.approx(np.std(x) / 2.5)
    assert fano_factor(x) == pytest.approx(np.var(x) / 2.5)


def test_gini_spans_zero_to_one():
    assert gini(np.ones(1000)) == pytest.approx(0.0, abs=1e-9)
    assert gini(np.array([0.0] * 999 + [1.0])) > 0.99


# --- autocorrelation and seasonality --------------------------------------

def test_acf_of_a_pure_tone_is_one_at_its_period():
    # The biased estimator shrinks every lag by (n - lag)/n = 8784/8928 = 0.9839,
    # so a perfectly periodic signal lands just under 1.0 rather than at it.
    assert acf_at(TONE, 144) == pytest.approx(1.0, abs=0.02)


def test_acf_of_white_noise_is_near_zero_at_any_lag():
    assert abs(acf_at(NOISE, 144)) < 0.05


def test_acf_ratio_is_high_for_a_seasonal_signal():
    """A strongly daily signal has lag-144 correlation comparable to lag-1."""
    assert acf_ratio(TONE, m=144) > 0.9


@pytest.mark.slow
def test_seasonal_strength_is_high_for_a_tone_and_low_for_noise():
    assert seasonal_strength(TONE + 0.05 * NOISE, period=144) > 0.9
    assert seasonal_strength(NOISE, period=144) < 0.5


@pytest.mark.slow
def test_profile_area_returns_every_documented_key():
    profile = profile_area(np.abs(TONE) * 100 + 10, period=144)
    assert set(profile) == {
        "mean", "std", "cv", "fano", "burstiness", "seasonal_strength",
        "perm_entropy", "spectral_entropy", "dfa_alpha", "acf_1", "acf_144", "acf_ratio",
    }
    assert all(np.isfinite(v) for v in profile.values())


