"""Tests for the v2 taxonomy vote and the document-level aggregator."""
from __future__ import annotations

import numpy as np
import pytest

from crc.agents.discipline.aggregate import (
    aggregate,
    chunk_weights,
    normalised_entropy,
    predict_document,
)
from crc.taxonomy import BY_CATEGORY, DISCIPLINES, EXCLUDED, label_paper


class TestTaxonomyMap:
    def test_no_category_both_mapped_and_excluded(self):
        assert not (set(BY_CATEGORY) & set(EXCLUDED))

    def test_every_mapping_targets_a_real_discipline(self):
        for m in BY_CATEGORY.values():
            assert m.discipline in DISCIPLINES
            if m.secondary:
                assert m.secondary in DISCIPLINES
                assert m.secondary != m.discipline

    def test_strengths_in_range(self):
        for m in BY_CATEGORY.values():
            assert 0.0 < m.strength <= 1.0

    def test_v1_error_entries_were_moved(self):
        # The four indefensible v1 entries measured at 38-100% error.
        assert BY_CATEGORY["cs.NE"].discipline == "Data Science"          # was IT
        assert BY_CATEGORY["cs.CY"].discipline == "Information Systems"   # was CS
        assert BY_CATEGORY["cs.HC"].discipline == "Information Systems"   # was CS
        assert BY_CATEGORY["cs.RO"].discipline == "Computer Engineering"  # was CS


class TestLabelVote:
    def test_unmapped_only_yields_no_label(self):
        r = label_paper(["math.PR", "q-bio.NC"])
        assert r.discipline is None
        assert not r.usable_for_training

    def test_primary_outweighs_secondary(self):
        # Same two categories, opposite order -> different winner. This is the
        # whole point of weighting the author-chosen primary.
        a = label_paper(["cs.DS", "cs.DC"])
        b = label_paper(["cs.DC", "cs.DS"])
        assert a.discipline == "Computer Science"
        assert b.discipline == "Information Technology"

    def test_clear_case_not_flagged_ambiguous(self):
        r = label_paper(["cs.LG", "stat.ML"])
        assert r.discipline == "Data Science"
        assert not r.ambiguous
        assert r.usable_for_training

    def test_genuinely_split_case_flagged(self):
        r = label_paper(["cs.AI", "cs.LG"])
        assert r.ambiguous
        assert not r.usable_for_training

    def test_margin_bounds(self):
        for cats in (["cs.SE"], ["cs.AI", "cs.LG"], ["cs.CR", "cs.NI"]):
            r = label_paper(cats)
            assert 0.0 <= r.margin <= 1.0


class TestAggregate:
    def _probs(self, rows):
        return np.array(rows, dtype=np.float64)

    def test_single_chunk_is_identity(self):
        # A pasted paragraph must pass straight through unchanged.
        p = self._probs([[0.7, 0.1, 0.05, 0.05, 0.05, 0.05]])
        for strat in ("mean", "weighted_mean", "weighted_geometric", "max"):
            doc, w = aggregate(p, ["abstract"], np.array([120.0]), strat)
            assert np.allclose(doc, p[0]), strat

    def test_output_is_a_distribution(self):
        rng = np.random.default_rng(0)
        raw = rng.random((7, 6))
        p = raw / raw.sum(axis=1, keepdims=True)
        secs = ["abstract", "methods", "results", "introduction",
                "other", "conclusion", "background"]
        nw = np.full(7, 180.0)
        for strat in ("mean", "weighted_mean", "weighted_geometric", "max"):
            doc, _ = aggregate(p, secs, nw, strat)
            assert pytest.approx(doc.sum(), abs=1e-9) == 1.0
            assert (doc >= 0).all()

    def test_uncertain_chunks_get_less_weight(self):
        # A near-uniform chunk carries almost no discipline signal and must not
        # outvote a confident one.
        confident = [0.9, 0.02, 0.02, 0.02, 0.02, 0.02]
        uniform = [1 / 6] * 6
        p = self._probs([confident, uniform])
        w = chunk_weights(["abstract", "abstract"], np.array([180.0, 180.0]), p)
        assert w[0] > w[1]

    def test_entropy_bounds(self):
        p = self._probs([[1 / 6] * 6, [1.0, 0, 0, 0, 0, 0]])
        e = normalised_entropy(np.clip(p, 1e-9, 1))
        assert pytest.approx(e[0], abs=1e-6) == 1.0
        assert e[1] < 0.01

    def test_longer_chunks_weigh_more_but_sublinearly(self):
        p = self._probs([[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]] * 2)
        w = chunk_weights(["methods", "methods"], np.array([100.0, 400.0]), p)
        assert w[1] > w[0]
        assert w[1] < 4 * w[0]          # sqrt, not linear

    def test_prediction_fields_consistent(self):
        p = self._probs([[0.6, 0.2, 0.05, 0.05, 0.05, 0.05],
                         [0.5, 0.3, 0.05, 0.05, 0.05, 0.05]])
        r = predict_document(p, ["abstract", "methods"], np.array([200.0, 200.0]))
        assert r.label in DISCIPLINES
        assert r.runner_up in DISCIPLINES
        assert r.label != r.runner_up
        assert r.gap >= 0
        assert pytest.approx(r.confidence, abs=1e-9) == float(r.probs.max())
        assert r.n_chunks == 2
