"""Feature Extraction module logic."""
import numpy as np
import pytest

from modules.feature_extraction import core


@pytest.fixture(scope="module")
def epochs(raw):
    return core.make_epochs(raw, ["left_hand", "right_hand"], 0.5, 3.5)


def test_band_power_finds_a_pure_tone():
    sf = 200.0
    t = np.arange(int(2 * sf)) / sf
    X = np.stack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 25 * t)])[None]  # 1 trial, 2 ch
    alpha, _, _ = core.band_power(X, sf, (8, 12), log=False)
    beta, _, _ = core.band_power(X, sf, (20, 30), log=False)
    assert alpha[0, 0] > 100 * alpha[0, 1]
    assert beta[0, 1] > 100 * beta[0, 0]


def test_hjorth_mobility_matches_frequency():
    sf = 500.0
    t = np.arange(int(2 * sf)) / sf
    X = np.stack([np.sin(2 * np.pi * 5 * t), np.sin(2 * np.pi * 20 * t)])[None]
    mob = core.hjorth(X, "mobility")[0]
    # mobility ≈ 2π f / sf for a sinusoid
    assert np.allclose(mob, 2 * np.pi * np.array([5, 20]) / sf, rtol=0.02)


def test_feature_shapes(epochs):
    n = len(epochs)
    bp = core.extract(epochs, "bandpower", (8, 13))
    assert bp.X.shape == (n, len(epochs.ch_names))
    csp = core.extract(epochs, "csp", (8, 13), n_csp=4)
    assert csp.X.shape == (n, 4)
    assert csp.extra["patterns"].shape == (4, len(epochs.ch_names))


def test_sensorimotor_features_rank_highest(epochs):
    r = core.extract(epochs, "bandpower", (8, 30))
    fs = core.fisher_score(r.X, r.y)
    top = {r.channel_of_feature[i] for i in np.argsort(fs)[::-1][:3]}
    assert top & {"C3", "C4"}


def test_cv_separates_task_but_not_precue_or_wrong_band(raw, epochs):
    good = core.cv_accuracy(epochs, "bandpower", (8, 30)).mean()
    assert good > core.chance_threshold(len(epochs), 2)
    pre = core.make_epochs(raw, ["left_hand", "right_hand"], -1.5, 0.0)
    assert core.cv_accuracy(pre, "bandpower", (8, 30)).mean() < 0.7


def test_csp_is_fitted_inside_folds(raw):
    """Leakage check: with shuffled labels, CV accuracy must stay near chance
    even though CSP fitted on *all* trials would separate them."""
    ep = core.make_epochs(raw, ["left_hand", "right_hand"], 0.5, 3.5, l_freq=8, h_freq=30)
    rng = np.random.default_rng(0)
    ep.events[:, 2] = rng.permutation(ep.events[:, 2])
    accs = [core.cv_accuracy(ep, "csp", (8, 30), seed=s).mean() for s in range(3)]
    assert np.mean(accs) < 0.7


def test_chance_threshold_known_values():
    assert core.chance_threshold(40, 2) == pytest.approx(0.625, abs=0.01)
    assert core.chance_threshold(1000, 2) < 0.53
