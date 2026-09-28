"""Tests for the prediction layer, including the ensemble wrapper.

`EnsembleClassifier` is production code on the inference path but is expensive
to exercise with real checkpoints, so it is tested here against stub members
that return fixed probabilities. That verifies the parts most likely to break —
weight normalisation, member combination, and the borderline contract — without
loading a model.
"""
from __future__ import annotations

import numpy as np
import pytest

from crc.agents.discipline.predict import (
    LOW_CONFIDENCE,
    TIGHT_GAP,
    ClassificationResult,
    EnsembleClassifier,
    borderline_check,
)
from crc.taxonomy import DISCIPLINES


class StubMember:
    """Stands in for a DisciplineClassifier: returns a fixed distribution."""

    def __init__(self, probs_row):
        self.row = np.array(probs_row, dtype=np.float64)
        self.calls = 0

    def chunk_probs(self, texts, batch_size=32):
        self.calls += 1
        return np.tile(self.row, (len(texts), 1))


class TestBorderlineCheck:
    def test_confident_and_separated_is_not_borderline(self):
        flag, reason = borderline_check(0.90, 0.60)
        assert flag is False
        assert reason is None

    def test_low_confidence_trips(self):
        flag, reason = borderline_check(LOW_CONFIDENCE - 0.05, 0.90)
        assert flag is True
        assert "LOW_CONFIDENCE" in reason

    def test_tight_gap_trips(self):
        flag, reason = borderline_check(0.95, TIGHT_GAP - 0.01)
        assert flag is True
        assert "TIGHT_GAP" in reason

    def test_thresholds_are_overridable(self):
        # Per-deployment tuning must be possible without editing the module.
        assert borderline_check(0.70, 0.50, low_confidence=0.80)[0] is True
        assert borderline_check(0.70, 0.50, low_confidence=0.60)[0] is False


class TestEnsembleClassifier:
    def _members(self):
        a = StubMember([0.6, 0.1, 0.1, 0.1, 0.05, 0.05])
        b = StubMember([0.1, 0.6, 0.1, 0.1, 0.05, 0.05])
        return a, b

    def test_rejects_empty_members(self):
        with pytest.raises(ValueError):
            EnsembleClassifier([])

    def test_weights_are_normalised(self):
        a, b = self._members()
        ens = EnsembleClassifier([a, b], weights=[3.0, 1.0])
        assert pytest.approx(ens.weights.sum(), abs=1e-9) == 1.0
        assert ens.weights[0] == pytest.approx(0.75)

    def test_equal_weights_by_default(self):
        a, b = self._members()
        ens = EnsembleClassifier([a, b])
        assert ens.weights == pytest.approx([0.5, 0.5])

    def test_combines_members_by_weight(self):
        a, b = self._members()
        ens = EnsembleClassifier([a, b], weights=[0.75, 0.25])
        out = ens.chunk_probs(["one chunk"])
        expected = 0.75 * a.row + 0.25 * b.row
        assert out.shape == (1, len(DISCIPLINES))
        assert np.allclose(out[0], expected)
        assert a.calls == 1 and b.calls == 1

    def test_output_rows_match_input_count(self):
        a, b = self._members()
        ens = EnsembleClassifier([a, b])
        assert ens.chunk_probs(["x", "y", "z"]).shape == (3, len(DISCIPLINES))

    def test_single_member_passthrough(self):
        a, _ = self._members()
        ens = EnsembleClassifier([a])
        assert np.allclose(ens.chunk_probs(["x"])[0], a.row)


class TestClassificationResult:
    def test_serialises_to_json(self):
        import json

        r = ClassificationResult(
            label="Computer Science", confidence=0.8, runner_up="Data Science",
            gap=0.5, probs={d: 1 / 6 for d in DISCIPLINES}, doc_type="paper",
            n_chunks=12, strategy="weighted_mean", borderline=False,
        )
        payload = json.loads(r.to_json())
        assert payload["label"] == "Computer Science"
        assert payload["n_chunks"] == 12
        assert set(payload["probs"]) == set(DISCIPLINES)
