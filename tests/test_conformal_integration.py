"""Tests for conformal prediction wired into the live inference path.

The load-bearing guarantee here is *architectural*, not statistical: the agents
must produce a complete answer on their own. A prediction set may say "this is
contested", but it must never be able to withhold the label, because no reviewer
or language model is present in the classification path to resolve it.
"""
from __future__ import annotations

import numpy as np

from crc.agents.discipline.conformal import (
    ConformalCalibration,
    calibrate,
    load_bank,
    pick_alpha,
    save_bank,
)
from crc.agents.discipline.predict import ClassificationResult
from crc.agents.field.predict import FieldClassifier
from crc.ingest import ingest
from crc.pipeline import Pipeline
from crc.taxonomy import DISCIPLINES


def _result(label, probs, borderline=False, pset=None):
    order = sorted(range(len(DISCIPLINES)), key=lambda i: -probs[i])
    return ClassificationResult(
        label=label,
        confidence=probs[DISCIPLINES.index(label)],
        runner_up=DISCIPLINES[order[1]],
        gap=0.1, probs=dict(zip(DISCIPLINES, probs)),
        doc_type="paragraph", n_chunks=1, strategy="weighted_mean",
        borderline=borderline, prediction_set=list(pset or []),
    )


#: qhat=0.0 admits only p >= 1.0, so every set comes out empty.
EMPTY = ConformalCalibration(method="lac", alpha=0.5,
                             class_conditional=False, qhat=0.0)
#: qhat=1.0 admits every label.
EVERYTHING = ConformalCalibration(method="lac", alpha=0.5,
                                  class_conditional=False, qhat=1.0)


class TestAnswerIsAlwaysCommitted:
    def test_empty_set_still_yields_the_top_label(self):
        pipe = Pipeline(None, None, shortlist=EMPTY)
        probs = [0.4, 0.3, 0.1, 0.1, 0.05, 0.05]
        res = _result(DISCIPLINES[0], probs)
        assert pipe._candidate_disciplines(res) == [DISCIPLINES[0]]

    def test_committed_label_always_leads_the_candidates(self):
        pipe = Pipeline(None, None, shortlist=EVERYTHING)
        probs = [0.1, 0.5, 0.2, 0.1, 0.05, 0.05]
        # top-1 by probability is DISCIPLINES[1]; the committed label leads even
        # though other candidates score higher.
        res = _result(DISCIPLINES[1], probs)
        cands = pipe._candidate_disciplines(res)
        assert cands[0] == DISCIPLINES[1]
        assert set(cands) == set(DISCIPLINES)
        # the rest are ordered by the model's own confidence
        rest = cands[1:]
        scores = [probs[DISCIPLINES.index(d)] for d in rest]
        assert scores == sorted(scores, reverse=True)

    def test_no_calibration_falls_back_to_top2_when_borderline(self):
        pipe = Pipeline(None, None, shortlist=None)
        probs = [0.4, 0.35, 0.1, 0.1, 0.03, 0.02]
        res = _result(DISCIPLINES[0], probs, borderline=True)
        assert pipe._candidate_disciplines(res) == [DISCIPLINES[0], DISCIPLINES[1]]

    def test_uncontested_answer_gets_no_extra_readings(self):
        pipe = Pipeline(None, None, shortlist=None)
        probs = [0.95, 0.02, 0.01, 0.01, 0.005, 0.005]
        res = _result(DISCIPLINES[0], probs, borderline=False)
        assert pipe._candidate_disciplines(res) == [DISCIPLINES[0]]

    def test_agent1_prediction_set_used_when_no_shortlist(self):
        pipe = Pipeline(None, None, shortlist=None)
        probs = [0.4, 0.3, 0.2, 0.05, 0.03, 0.02]
        res = _result(DISCIPLINES[0], probs,
                      pset=[DISCIPLINES[0], DISCIPLINES[2]])
        assert pipe._candidate_disciplines(res) == [DISCIPLINES[0], DISCIPLINES[2]]


class TestSingleEncoderPass:
    def test_many_disciplines_cost_one_pass(self):
        """Conditioning is a mask, so the shortlist must not re-encode."""
        clf = object.__new__(FieldClassifier)
        calls = {"n": 0}

        def fake_chunk_logits(texts, batch_size=32):
            calls["n"] += 1
            rng = np.random.default_rng(0)
            return rng.normal(size=(max(len(texts), 1), 38))

        clf.chunk_logits = fake_chunk_logits
        doc = ingest("A study of distributed consensus protocols under churn.")

        out = clf.predict_many(doc, list(DISCIPLINES))
        assert len(out) == len(DISCIPLINES)
        assert calls["n"] == 1, f"re-encoded {calls['n']} times"
        # each reading must stay inside its own discipline's ballot
        for d, r in zip(DISCIPLINES, out):
            assert r.discipline == d

    def test_empty_candidate_list_does_no_work(self):
        clf = object.__new__(FieldClassifier)
        calls = {"n": 0}
        clf.chunk_logits = lambda t, batch_size=32: (
            calls.__setitem__("n", calls["n"] + 1) or np.zeros((1, 38)))
        doc = ingest("Short note.")
        assert clf.predict_many(doc, []) == []
        assert calls["n"] == 0


class TestBankRoundTrip:
    def test_bank_holds_many_alphas_and_picks_nearest(self, tmp_path):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(3000, len(DISCIPLINES))) * 2
        e = np.exp(logits - logits.max(1, keepdims=True))
        probs = e / e.sum(1, keepdims=True)
        labels = np.array([rng.choice(len(DISCIPLINES), p=p) for p in probs])

        bank = {a: calibrate(probs, labels, alpha=a, class_conditional=True)
                for a in (0.05, 0.10, 0.20)}
        p = tmp_path / "conformal.json"
        save_bank(p, bank)

        back = load_bank(p)
        assert set(back) == {0.05, 0.10, 0.20}
        assert pick_alpha(back, 0.10).alpha == 0.10
        # an uncalibrated alpha resolves to the closest rather than failing
        assert pick_alpha(back, 0.11).alpha == 0.10
        assert pick_alpha(back, 0.30).alpha == 0.20

    def test_legacy_single_alpha_file_still_loads(self, tmp_path):
        cal = ConformalCalibration(method="lac", alpha=0.15,
                                   class_conditional=False, qhat=0.5)
        p = tmp_path / "conformal.json"
        cal.save(p)
        back = load_bank(p)
        assert list(back) == [0.15]
