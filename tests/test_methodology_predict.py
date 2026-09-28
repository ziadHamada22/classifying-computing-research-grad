"""Tests for Agent 3 deployment (facets, and the design derived from them).

The properties pinned here are about auditability and honesty rather than
accuracy: the derived design must always be traceable to the rule that produced
it, facets that were not observed must not be asserted, papers that are genuinely
several things at once must be reported as such, and a document with no abstract
must still get an answer.
"""
from __future__ import annotations

import numpy as np

from crc.agents.methodology.predict import (
    FACET_THRESHOLD,
    READ_ORDER,
    MethodologyClassifier,
    rule_probability,
)
from crc.ingest import ingest
from crc.ingest.schema import Section
from crc.taxonomy.facets import (
    DESIGN_RULES,
    FACET_KEYS,
    derive_designs,
    primary_design,
)
from crc.taxonomy.methodology import DESIGNS

NL = chr(10)


def _stub(logits_row=None):
    """A classifier with a fake encoder, so tests need no GPU and no weights."""
    clf = object.__new__(MethodologyClassifier)
    if logits_row is None:
        rng = np.random.default_rng(0)
        clf.facet_probs = lambda texts, batch_size=16: rng.random(
            (max(len(texts), 1), len(FACET_KEYS)))
    else:
        row = np.asarray(logits_row, dtype=float)[None, :]
        clf.facet_probs = lambda texts, batch_size=16: np.repeat(
            row, max(len(texts), 1), axis=0)
    return clf


def _facets(**kw):
    return {f: bool(kw.get(f, False)) for f in FACET_KEYS}


class TestDerivationIsAuditable:
    def test_every_answer_names_the_rule_that_produced_it(self):
        r = _stub().predict(ingest("We implement and release a new compiler."))
        assert r.derived_rule
        assert any("derived from facets by the rule" in n for n in r.notes)

    def test_reported_design_matches_the_taxonomy_derivation(self):
        r = _stub().predict(ingest("We prove a lower bound on consensus rounds."))
        hard = {f: v >= FACET_THRESHOLD for f, v in r.facets.items()}
        assert r.design == primary_design(hard)

    def test_all_nine_designs_are_reachable(self):
        # The previous model could only ever return four. Every design named in
        # the taxonomy must be produced by some facet combination.
        reachable = {d for d, _ in DESIGN_RULES}
        assert reachable == {d.name for d in DESIGNS}

    def test_rule_probability_uses_negation_correctly(self):
        p = {f: 0.0 for f in FACET_KEYS}
        p["proves"] = 0.9
        p["builds"] = 0.2
        # "proves and not builds" -> 0.9 * 0.8
        assert abs(rule_probability("proves and not builds", p) - 0.72) < 1e-9

    def test_default_rule_has_no_confidence(self):
        assert rule_probability("default (no facet fired)", {}) == 0.0


class TestMultiFacetHonesty:
    def test_a_paper_can_be_two_things_at_once(self):
        both = derive_designs(_facets(builds=True, evaluates=True))
        assert len(both) > 1, "build+evaluate must not collapse to one design"

    def test_underdetermined_answers_are_flagged(self):
        row = np.zeros(len(FACET_KEYS))
        row[FACET_KEYS.index("builds")] = 0.9
        row[FACET_KEYS.index("evaluates")] = 0.9
        r = _stub(row).predict(ingest("We implement a system and evaluate it on "
                                      "four benchmark datasets."))
        assert len(r.compatible_designs) > 1
        assert r.borderline
        assert any("more than one design" in n for n in r.notes)

    def test_absent_facets_are_not_asserted(self):
        row = np.zeros(len(FACET_KEYS)) + 0.01
        r = _stub(row).predict(ingest("A short note about nothing in particular."))
        assert r.facets_present == []
        assert any("no positive evidence" in n for n in r.notes)

    def test_facet_probabilities_are_all_reported(self):
        r = _stub().predict(ingest("We simulate urban traffic for 10,000 hours."))
        assert set(r.facets) == set(FACET_KEYS)
        assert all(0.0 <= v <= 1.0 for v in r.facets.values())


class TestAlwaysAnswers:
    def test_pasted_text_is_treated_as_an_abstract_without_a_warning(self):
        r = _stub().predict(ingest("We evaluate six retrieval models on four "
                                   "benchmark datasets with significance tests."))
        assert r.design
        assert not any("more caution" in n for n in r.notes)

    def test_structured_document_is_read(self):
        body = ("Deep Learning for X" + NL + NL + "Abstract" + NL + NL
                + "word " * 80 + NL + NL + "1 Introduction" + NL + NL
                + "intro " * 120)
        r = _stub().predict(ingest(body))
        assert r.n_blocks >= 1
        assert r.design

    def test_worldview_and_method_are_derived_not_invented(self):
        from crc.taxonomy.methodology import method_of, worldview_of

        r = _stub().predict(ingest("We implement and release a new compiler."))
        assert r.worldview == worldview_of(r.design)
        assert r.method == method_of(r.design)


class TestReadOrder:
    def test_abstract_is_preferred_over_methods(self):
        assert READ_ORDER[0] == Section.ABSTRACT.value
        assert (READ_ORDER.index(Section.ABSTRACT.value)
                < READ_ORDER.index(Section.METHODS.value))
