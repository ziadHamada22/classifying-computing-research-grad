"""Tests for the end-to-end pipeline eval helper.

The load-bearing piece is ``conditioned_field_pred``: it must pick the best field
*within a given discipline's ballot*, which is exactly the masking that makes the
oracle-vs-predicted comparison meaningful. If it ever picked a field outside the
discipline, the cascade identity would silently break.
"""
from __future__ import annotations

import numpy as np

from crc.eval.evaluate_pipeline import conditioned_field_pred
from crc.taxonomy import DISCIPLINES
from crc.taxonomy.fields import DISCIPLINE_FIELD_IDS, N_GLOBAL_FIELDS


class TestConditionedFieldPred:
    def test_prediction_stays_on_the_ballot(self):
        # A logit vector that peaks on a field of the WRONG discipline must never
        # be returned when conditioning on the right one.
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(50, N_GLOBAL_FIELDS))
        for d in DISCIPLINES:
            preds = conditioned_field_pred(logits, [d] * 50)
            allowed = set(DISCIPLINE_FIELD_IDS[d])
            assert all(int(p) in allowed for p in preds)

    def test_picks_the_argmax_within_discipline(self):
        d = DISCIPLINES[0]
        allowed = DISCIPLINE_FIELD_IDS[d]
        logits = np.full((1, N_GLOBAL_FIELDS), -9.0)
        # make a non-first allowed field the clear winner
        target = allowed[len(allowed) // 2]
        logits[0, target] = 5.0
        # and put an even bigger spike OUTSIDE the discipline to try to fool it
        outside = next(i for i in range(N_GLOBAL_FIELDS) if i not in set(allowed))
        logits[0, outside] = 9.0
        assert int(conditioned_field_pred(logits, [d])[0]) == target

    def test_oracle_recovers_true_label_when_confident(self):
        # If every paper's logits spike on its own true field, conditioning on the
        # true discipline recovers 100% — the sanity floor the oracle rests on.
        labels, discs = [], []
        logits_rows = []
        for d in DISCIPLINES:
            for fid in DISCIPLINE_FIELD_IDS[d]:
                row = np.full(N_GLOBAL_FIELDS, -9.0)
                row[fid] = 9.0
                logits_rows.append(row)
                labels.append(fid)
                discs.append(d)
        preds = conditioned_field_pred(np.array(logits_rows), discs)
        assert (preds == np.array(labels)).all()
