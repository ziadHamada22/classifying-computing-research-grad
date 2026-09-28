"""Tests for conformal prediction.

The coverage guarantee is the whole point of this module, and it is the kind of
claim that can be quietly wrong (an off-by-one in the quantile level still
"works" and still produces plausible sets). So the central test is empirical:
build synthetic exchangeable data, split it, calibrate on one half, and check
coverage on the other actually lands at 1 - alpha.
"""
from __future__ import annotations

import numpy as np

from crc.agents.discipline.conformal import (
    ConformalCalibration,
    aps_score_matrix,
    calibrate,
    conformal_quantile,
    evaluate_sets,
    lac_score_matrix,
)
from crc.taxonomy import DISCIPLINES

K = len(DISCIPLINES)


def _synthetic(n: int, k: int = K, sharpness: float = 3.0, seed: int = 0):
    """Exchangeable (probs, labels): a plausibly-calibrated soft classifier."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, k)) * sharpness
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    # Draw the label FROM the predicted distribution, so probs are calibrated by
    # construction and coverage should track alpha closely.
    labels = np.array([rng.choice(k, p=p) for p in probs])
    return probs, labels


class TestConformalQuantile:
    def test_uses_finite_sample_corrected_level(self):
        # n=9, alpha=0.1 -> ceil(10*0.9)/9 = 9/9 = 1.0 -> the maximum score.
        s = np.arange(9, dtype=float)
        assert conformal_quantile(s, 0.1) == 8.0

    def test_returns_inf_when_calibration_too_small(self):
        # n=8, alpha=0.1 -> ceil(9*0.9)/8 = 9/8 > 1 -> cannot guarantee.
        assert conformal_quantile(np.arange(8, dtype=float), 0.1) == float("inf")
        assert conformal_quantile(np.array([]), 0.1) == float("inf")

    def test_monotone_in_alpha(self):
        s = np.linspace(0, 1, 500)
        assert (conformal_quantile(s, 0.20) <= conformal_quantile(s, 0.10)
                <= conformal_quantile(s, 0.05))


class TestScoreMatrices:
    def test_lac_is_one_minus_prob(self):
        p = np.array([[0.7, 0.2, 0.1]])
        assert np.allclose(lac_score_matrix(p), [[0.3, 0.8, 0.9]])

    def test_aps_deterministic_top_class_scores_its_own_mass(self):
        p = np.array([[0.5, 0.3, 0.2]])
        s = aps_score_matrix(p)
        assert np.isclose(s[0, 0], 0.5)          # rank 1: own mass
        assert np.isclose(s[0, 1], 0.8)          # rank 2: 0.5 + 0.3
        assert np.isclose(s[0, 2], 1.0)          # rank 3: all mass

    def test_aps_is_monotone_in_rank(self):
        probs, _ = _synthetic(200, seed=1)
        s = aps_score_matrix(probs)
        for i in range(len(probs)):
            order = np.argsort(-probs[i])
            assert np.all(np.diff(s[i][order]) >= -1e-12)

    def test_aps_randomised_stays_below_deterministic(self):
        probs, _ = _synthetic(200, seed=2)
        rng = np.random.default_rng(0)
        assert np.all(aps_score_matrix(probs, rng) <= aps_score_matrix(probs) + 1e-12)


class TestCoverageGuarantee:
    def test_lac_marginal_coverage_holds(self):
        probs, labels = _synthetic(20_000, seed=3)
        cal = calibrate(probs[:10_000], labels[:10_000], alpha=0.1, method="lac")
        r = evaluate_sets(cal, probs[10_000:], labels[10_000:])
        # Guarantee is >= 0.9; allow a small finite-sample slack downward.
        assert r["coverage"] >= 0.885, r["coverage"]
        assert r["coverage"] <= 0.94, r["coverage"]   # and not absurdly conservative

    def test_aps_marginal_coverage_holds(self):
        probs, labels = _synthetic(20_000, seed=4)
        cal = calibrate(probs[:10_000], labels[:10_000], alpha=0.1, method="aps")
        r = evaluate_sets(cal, probs[10_000:], labels[10_000:])
        assert r["coverage"] >= 0.885, r["coverage"]

    def test_coverage_tracks_several_alphas(self):
        probs, labels = _synthetic(20_000, seed=5)
        for alpha in (0.05, 0.10, 0.20):
            cal = calibrate(probs[:10_000], labels[:10_000], alpha=alpha)
            r = evaluate_sets(cal, probs[10_000:], labels[10_000:])
            assert r["coverage"] >= (1 - alpha) - 0.02, (alpha, r["coverage"])

    def test_class_conditional_covers_every_class(self):
        probs, labels = _synthetic(30_000, seed=6)
        cal = calibrate(probs[:15_000], labels[:15_000], alpha=0.1,
                        class_conditional=True)
        r = evaluate_sets(cal, probs[15_000:], labels[15_000:])
        for name, s in r["per_class"].items():
            assert s["coverage"] >= 0.86, (name, s["coverage"])

    def test_lac_sets_are_no_larger_than_aps(self):
        # LAC is the minimum-average-size rule; this is its defining property.
        probs, labels = _synthetic(20_000, seed=7)
        tr, te = slice(0, 10_000), slice(10_000, None)
        lac = evaluate_sets(calibrate(probs[tr], labels[tr], method="lac"),
                            probs[te], labels[te])
        aps = evaluate_sets(calibrate(probs[tr], labels[tr], method="aps"),
                            probs[te], labels[te])
        assert lac["avg_set_size"] <= aps["avg_set_size"] + 1e-9


class TestSetConstruction:
    def test_lac_set_is_prob_above_one_minus_qhat(self):
        cal = ConformalCalibration(method="lac", alpha=0.1,
                                   class_conditional=False, qhat=0.6)
        p = np.array([[0.5, 0.3, 0.15, 0.05]])
        mask = cal.predict_set(p)
        # included iff p >= 1 - 0.6 = 0.4
        assert mask.tolist() == [[True, False, False, False]]

    def test_larger_alpha_gives_smaller_sets(self):
        probs, labels = _synthetic(8_000, seed=8)
        sizes = []
        for alpha in (0.02, 0.10, 0.30):
            cal = calibrate(probs[:4_000], labels[:4_000], alpha=alpha)
            sizes.append(evaluate_sets(cal, probs[4_000:],
                                       labels[4_000:])["avg_set_size"])
        assert sizes[0] >= sizes[1] >= sizes[2]

    def test_route_flags_non_singletons(self):
        cal = ConformalCalibration(method="lac", alpha=0.1,
                                   class_conditional=False, qhat=0.5)
        # row 0: one class above 0.5 -> singleton -> commit
        # row 1: two classes above 0.5 -> route
        p = np.array([[0.9, 0.05, 0.05], [0.5, 0.5, 0.0]])
        assert cal.route(p).tolist() == [False, True]

    def test_set_labels_are_discipline_names(self):
        cal = ConformalCalibration(method="lac", alpha=0.1,
                                   class_conditional=False, qhat=0.9)
        p = np.zeros((1, K))
        p[0, 0] = 0.6
        p[0, 1] = 0.4
        names = cal.set_labels(p)[0]
        assert set(names) <= set(DISCIPLINES)
        assert DISCIPLINES[0] in names and DISCIPLINES[1] in names


class TestPersistence:
    def test_round_trip_preserves_behaviour(self, tmp_path):
        probs, labels = _synthetic(2_000, seed=9)
        cal = calibrate(probs, labels, alpha=0.1, class_conditional=True,
                        calib_source="unit-test")
        p = tmp_path / "conformal.json"
        cal.save(p)
        back = ConformalCalibration.load(p)
        assert back.calib_source == "unit-test"
        assert np.array_equal(cal.predict_set(probs), back.predict_set(probs))
