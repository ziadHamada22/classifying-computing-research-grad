"""Tests for the Agent 2 completion work: conformal field sets, the tuned
fallback flag, and the ballot arithmetic the evaluations rely on.

No checkpoint is loaded: the field model's encoder is stubbed, so these pin the
decision logic, not accuracy.
"""
from __future__ import annotations

import numpy as np
import pytest

from crc.agents.discipline.conformal import calibrate
from crc.agents.field.predict import (
    LEGACY_LOW_CONFIDENCE,
    LEGACY_TIGHT_GAP,
    LOW_CONFIDENCE,
    TIGHT_GAP,
    FieldClassifier,
)
from crc.ingest import ingest
from crc.taxonomy import DISCIPLINES
from crc.taxonomy.fields import DISCIPLINE_FIELD_IDS, GLOBAL_ID2LABEL

N = len(GLOBAL_ID2LABEL)
CS = DISCIPLINES[0]


def stub_classifier(logit_row: np.ndarray, conformal=None) -> FieldClassifier:
    clf = object.__new__(FieldClassifier)
    clf.chunk_logits = lambda texts, batch_size=32: np.tile(logit_row, (max(len(texts), 1), 1))
    clf.conformal = conformal
    return clf


def fitted_bank(alpha: float = 0.1):
    """A marginal LAC predictor fitted on synthetic ballot-masked probabilities."""
    rng = np.random.default_rng(0)
    allowed = DISCIPLINE_FIELD_IDS[CS]
    P = np.zeros((400, N))
    y = rng.choice(allowed, size=400)
    for i, t in enumerate(y):
        # true field gets 0.25 plus its random share: probabilities spread from
        # ~0.25 to ~1, like a real model's, so the fitted bar sits well below 0.5
        p = 0.75 * rng.dirichlet(np.ones(len(allowed)))
        p[allowed.index(t)] += 0.25
        P[i, allowed] = p
    return calibrate(P, y, alpha=alpha, method="lac")


def doc():
    return ingest("We study consensus protocols for distributed systems under churn.")


class TestConformalFieldSets:
    def test_confident_answer_is_a_certified_singleton(self):
        row = np.full(N, -5.0)
        row[DISCIPLINE_FIELD_IDS[CS][2]] = 8.0
        r = stub_classifier(row, fitted_bank()).predict(doc(), CS)
        assert r.prediction_set == [r.label]
        assert r.certified and not r.borderline and r.set_size == 1

    def test_tie_is_contested_and_names_both_fields(self):
        row = np.full(N, -5.0)
        a, b = DISCIPLINE_FIELD_IDS[CS][:2]
        row[a] = row[b] = 3.0
        r = stub_classifier(row, fitted_bank()).predict(doc(), CS)
        assert r.borderline and r.set_size >= 2
        assert {GLOBAL_ID2LABEL[a], GLOBAL_ID2LABEL[b]} <= set(r.prediction_set)
        assert "CONFORMAL" in r.borderline_reason

    def test_set_never_leaves_the_ballot(self):
        row = np.random.default_rng(1).normal(size=N)
        r = stub_classifier(row, fitted_bank(0.01)).predict(doc(), CS)
        ballot = {GLOBAL_ID2LABEL[i] for i in DISCIPLINE_FIELD_IDS[CS]}
        assert set(r.prediction_set) <= ballot
        assert r.label in ballot

    def test_set_is_serialised(self):
        row = np.zeros(N)
        r = stub_classifier(row, fitted_bank()).predict(doc(), CS)
        assert '"prediction_set"' in r.to_json()


class TestFallbackFlag:
    def test_tuned_thresholds_are_the_defaults(self):
        assert (LOW_CONFIDENCE, TIGHT_GAP) == (0.80, 0.02)
        assert (LEGACY_LOW_CONFIDENCE, LEGACY_TIGHT_GAP) == (0.55, 0.12)

    def test_without_a_bank_the_tuned_flag_applies(self):
        # top field ~0.70 of the ballot: flagged under 0.80, fine under legacy 0.55
        allowed = DISCIPLINE_FIELD_IDS[CS]
        row = np.full(N, -20.0)
        row[allowed] = 0.0
        row[allowed[0]] = np.log(0.70 / 0.05)       # rest share 0.30 over 6 fields
        r = stub_classifier(row).predict(doc(), CS)
        assert r.prediction_set == [] and r.conformal_alpha is None
        assert r.borderline and "LOW_CONFIDENCE" in r.borderline_reason


class TestBallotArithmetic:
    def test_ballot_matrix_is_a_distribution_on_the_ballot(self):
        from crc.eval.evaluate_field_uncertainty import ballot_matrix
        rng = np.random.default_rng(2)
        logits = rng.normal(size=(len(DISCIPLINES), N))
        P = ballot_matrix(logits, list(DISCIPLINES))
        np.testing.assert_allclose(P.sum(axis=1), 1.0)
        for i, d in enumerate(DISCIPLINES):
            off = np.setdiff1d(np.arange(N), DISCIPLINE_FIELD_IDS[d])
            assert (P[i, off] == 0).all()

    def test_conditioned_baseline_picks_inside_the_ballot(self):
        from crc.agents.field.train_baseline import conditioned
        classes = np.arange(N)
        scores = np.zeros((1, N))
        scores[0, DISCIPLINE_FIELD_IDS[DISCIPLINES[1]][0]] = 10.0   # an IS field
        pick = conditioned(scores, classes, [CS])[0]
        assert pick in DISCIPLINE_FIELD_IDS[CS]


class TestAbstractPooling:
    def _paper(self):
        from crc.ingest.schema import Chunk, Document, DocType, Section
        parts = [("A Paper Title", Section.TITLE),
                 ("We study consensus protocols.", Section.ABSTRACT),
                 ("Related work on Paxos.", Section.RELATED_WORK),
                 ("Our results show ...", Section.RESULTS)]
        chunks = [Chunk(text=t, section=s, index=i, n_words=len(t.split()))
                  for i, (t, s) in enumerate(parts)]
        return Document(text=" ".join(t for t, _ in parts), doc_type=DocType.PAPER,
                        title="A Paper Title", chunks=chunks)

    def test_a_paper_is_read_through_its_abstract_only(self):
        seen = []
        clf = object.__new__(FieldClassifier)
        clf.chunk_logits = lambda texts, batch_size=32: seen.append(list(texts)) or np.zeros((len(texts), N))
        clf.predict(self._paper(), CS)
        assert len(seen) == 1 and len(seen[0]) == 1
        assert seen[0][0].startswith("[abstract]")

    def test_mean_pooling_reads_every_chunk(self):
        seen = []
        clf = object.__new__(FieldClassifier)
        clf.pooling = "mean"
        clf.chunk_logits = lambda texts, batch_size=32: seen.append(list(texts)) or np.zeros((len(texts), N))
        clf.predict(self._paper(), CS)
        assert len(seen[0]) == 4

    def test_text_without_an_abstract_uses_all_chunks(self):
        clf = object.__new__(FieldClassifier)
        chunks = clf.evidence_chunks(doc())      # a pasted paragraph: no abstract
        assert len(chunks) >= 1
