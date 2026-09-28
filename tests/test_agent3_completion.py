"""Tests for the September Agent 3 work: vocabulary v2, per-facet thresholds,
the reference store, the no-evidence fallback and the external arXiv signals.

What is pinned is behaviour a later edit could silently undo: the legacy yardstick
must stay reproducible, the broad "we propose a method" rule must stay out, a paper
with no evidence must never be passed off as a facet-derived answer, and neighbours
without evidence must not vote the old default back in.
"""
from __future__ import annotations

import numpy as np
import pytest

from crc.agents.methodology.predict import (
    N_SIMILAR,
    MethodologyClassifier,
)
from crc.agents.methodology.retrieval import DEFAULT_DESIGN, ReferenceStore
from crc.data.external_signals import SIGNAL_FACET, SIGNALS, detect
from crc.data.label_facets import ABSENT, PRESENT, vote_region
from crc.eval.evaluate_facet_decisions import fit_thresholds
from crc.ingest import ingest
from crc.taxonomy.facets import FACET_KEYS, extended_evidence

PAD = " The remaining sentences only pad this region past the minimum length." * 8


# --------------------------------------------------------------- vocabulary v2
class TestVocabularyV2:
    def test_named_artefact_counts_as_building(self):
        t = "We introduce Wasm-Mutate, a compiler-agnostic diversification engine."
        assert extended_evidence("builds", t)

    def test_broad_propose_rule_stays_rejected(self):
        # Measured and rejected: it doubled builds and pushed Design & Creation
        # to 55%. A generic "we propose a method" is not v2 builds evidence.
        assert not extended_evidence("builds", "We propose a method to estimate the t-statistic.")
        assert not extended_evidence("builds", "We present an empirical study of code review.")

    def test_quantified_gain_counts_as_evaluation(self):
        assert extended_evidence("evaluates", "Our approach improves accuracy by 12.5% over prior work.")
        assert extended_evidence("evaluates", "Experiments on three datasets show consistent results.")
        assert not extended_evidence("evaluates", "About 12% of the population is left-handed.")

    def test_bounds_count_as_proof(self):
        assert extended_evidence("proves", "The master algorithm enjoys O(d log T) regret bounds.")
        assert extended_evidence("proves", "The problem admits a polynomial-time algorithm.")
        assert not extended_evidence("proves", "Oscar(2019) studied the problem.")

    def test_legacy_yardstick_is_reproducible(self):
        t = "We introduce Wasm-Mutate, a compiler-agnostic engine." + PAD
        n = len(t.split())
        assert vote_region(t, "abstract", n, extended=True)["builds"] == PRESENT
        assert vote_region(t, "abstract", n, extended=False)["builds"] == ABSENT

    def test_only_three_facets_were_extended(self):
        t = "We introduce FooBar, a tool; experiments on data show gains; O(n) bound."
        others = [f for f in FACET_KEYS if f not in ("builds", "evaluates", "proves")]
        assert not any(extended_evidence(f, t) for f in others)


# ------------------------------------------------------------- reference store
def _store():
    # four reference papers on a 3-d sphere; the last has no facet at all
    emb = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0.95, 0, 0.05]], np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    facets = np.zeros((4, len(FACET_KEYS)), bool)
    facets[0, FACET_KEYS.index("proves")] = True
    facets[1, FACET_KEYS.index("proves")] = True
    facets[2, FACET_KEYS.index("evaluates")] = True
    designs = ["Formal / Theoretical", "Formal / Theoretical",
               "Experiment / Empirical Evaluation", DEFAULT_DESIGN]
    return ReferenceStore(emb, ["a", "b", "c", "d"], ["A", "B", "C", "D"], designs, facets)


class TestReferenceStore:
    def test_search_is_sorted_by_similarity(self):
        idx, sims = _store().search(np.array([1, 0, 0]), k=4)
        assert list(sims) == sorted(sims, reverse=True)
        assert idx[0] == 0

    def test_neighbours_report_design_and_facets(self):
        n = _store().neighbours(np.array([0, 1, 0]), k=1)[0].to_dict()
        assert n["paper_id"] == "c" and n["design"].startswith("Experiment")
        assert n["facets"] == ["evaluates"]

    def test_neighbours_without_evidence_do_not_vote(self):
        # "d" is the closest to this query but had no facet: its design is the
        # old default and must not be laundered back in through the vote.
        design, share, vote = _store().design_vote(np.array([0.96, 0, 0.04]), k=2)
        assert design == "Formal / Theoretical"
        assert DEFAULT_DESIGN not in vote

    def test_store_with_no_evidence_falls_back_to_default(self):
        s = _store()
        s.has_evidence[:] = False
        assert s.design_vote(np.array([1, 0, 0]))[:2] == (DEFAULT_DESIGN, 0.0)

    def test_save_load_roundtrip(self, tmp_path):
        s = _store()
        s.save(tmp_path / "s.npz")
        r = ReferenceStore.load(tmp_path / "s.npz")
        assert list(r.paper_ids) == list(s.paper_ids)
        assert np.allclose(r.emb, s.emb, atol=1e-3)
        assert (r.facets == s.facets).all()


# ------------------------------------------------ the classifier's decisions
def _clf(row, store=None, fallback=None, thresholds=None):
    clf = object.__new__(MethodologyClassifier)
    probs = np.asarray(row, float)[None, :]
    emb = np.array([[0.96, 0.0, 0.04]], np.float32)
    clf.encode = lambda texts, batch_size=16: (np.repeat(probs, len(texts), 0),
                                               np.repeat(emb, len(texts), 0))
    clf.facet_probs = lambda texts, batch_size=16: np.repeat(probs, len(texts), 0)
    from crc.agents.methodology.predict import normalise_fallback
    clf.store, clf.thresholds = store, thresholds
    clf.no_evidence_fallback = normalise_fallback(fallback, store)
    return clf


NOTHING = [0.2] * len(FACET_KEYS)


class TestNoEvidence:
    def test_default_is_flagged_not_passed_off(self):
        r = _clf(NOTHING).predict(ingest("An essay on the future of computing."))
        assert r.design_source == "default"
        assert r.borderline and r.borderline_reason.startswith("NO_EVIDENCE")
        assert r.confidence == 0.0

    def test_fallback_uses_similar_papers_and_says_so(self):
        r = _clf(NOTHING, store=_store(), fallback="similar_papers").predict(
            ingest("An essay on the future of computing."))
        assert r.design_source == "similar_papers"
        assert r.design == "Formal / Theoretical"
        assert "similar reference papers" in r.derived_rule
        assert r.borderline and "similar reference papers" in r.borderline_reason

    def test_fallback_is_off_unless_enabled(self):
        r = _clf(NOTHING, store=_store(), fallback=None).predict(ingest("An essay."))
        assert r.design_source == "default"

    def test_top_facet_fallback_names_the_facet_it_used(self):
        row = list(NOTHING)
        row[FACET_KEYS.index("proves")] = 0.41
        r = _clf(row, fallback="top_facet").predict(ingest("An essay on bounds."))
        assert r.design_source == "top_facet"
        assert r.design == "Formal / Theoretical"
        assert "proves" in r.derived_rule and "0.41" in r.derived_rule
        assert r.borderline and r.borderline_reason.startswith("NO_EVIDENCE")
        assert r.confidence == pytest.approx(0.41)

    def test_similar_papers_mode_needs_a_store(self):
        from crc.agents.methodology.predict import normalise_fallback
        assert normalise_fallback("similar_papers", None) is None
        assert normalise_fallback(True, _store()) == "similar_papers"
        assert normalise_fallback("top_facet", None) == "top_facet"
        assert normalise_fallback("nonsense", _store()) is None

    def test_facet_evidence_is_never_overridden_by_neighbours(self):
        row = list(NOTHING)
        row[FACET_KEYS.index("evaluates")] = 0.9
        r = _clf(row, store=_store(), fallback="similar_papers").predict(ingest("We evaluate X."))
        assert r.design_source == "facets"
        assert r.design.startswith("Experiment")

    def test_similar_papers_are_reported_with_a_store(self):
        r = _clf(NOTHING, store=_store()).predict(ingest("Some abstract."))
        assert len(r.similar_papers) == min(N_SIMILAR, 4)
        assert {"paper_id", "title", "design", "facets", "similarity"} <= set(r.similar_papers[0])

    def test_no_store_means_no_similar_papers(self):
        assert _clf(NOTHING).predict(ingest("Some abstract.")).similar_papers == []


class TestThresholds:
    def test_fitted_threshold_changes_the_decision(self):
        row = list(NOTHING)
        row[FACET_KEYS.index("secondary")] = 0.3
        r0 = _clf(row).predict(ingest("A review."))
        r1 = _clf(row, thresholds={"secondary": 0.25}).predict(ingest("A review."))
        assert "secondary" not in r0.facets_present
        assert "secondary" in r1.facets_present
        assert r1.design.startswith("Systematic Literature Review")

    def test_fit_thresholds_recovers_a_shifted_boundary(self):
        rng = np.random.default_rng(0)
        soft = rng.random((2000, len(FACET_KEYS)))
        probs = np.clip(soft - 0.2, 0, 1)        # the model reads 0.2 low everywhere
        thr = fit_thresholds(probs, soft)
        assert all(abs(t - 0.3) <= 0.02 for t in thr.values())


# ------------------------------------------------------------ external signals
class TestExternalSignals:
    def test_theory_venue(self):
        assert detect("Accepted to STOC 2024", None)["theory_venue"]
        assert detect(None, "Theoretical Computer Science 950 (2023)")["theory_venue"]

    def test_acronyms_are_case_sensitive(self):
        assert not detect("we used the chi-square test", None)["hci_venue"]
        assert detect("CHI 2024, Honolulu", None)["hci_venue"]

    def test_code_release(self):
        assert detect("Code available at https://github.com/x/y", None)["code_release"]

    def test_under_review_is_not_a_survey(self):
        assert not detect("Currently under review", None)["survey_venue"]

    def test_every_signal_but_position_names_a_facet(self):
        assert set(SIGNAL_FACET) == set(SIGNALS) - {"position"}
        assert set(SIGNAL_FACET.values()) <= set(FACET_KEYS)


@pytest.mark.parametrize("key", ["builds", "evaluates", "proves"])
def test_extended_evidence_handles_empty_text(key):
    assert extended_evidence(key, "") is False


# ------------------------------------------------ the proposal's RAG few-shot LLM
class TestLLMDesign:
    def test_rubric_names_every_design(self):
        from crc.agents.methodology.llm_design import design_rubric
        from crc.taxonomy.methodology import DESIGN_NAMES

        r = design_rubric()
        assert all(d in r for d in DESIGN_NAMES)

    def test_examples_skip_no_evidence_and_the_paper_itself(self):
        from crc.agents.methodology.llm_design import retrieved_examples

        s = _store()                       # "d" has no facet; "a" is the query itself
        texts = {p: f"text {p}" for p in "abcd"}
        ex = retrieved_examples(s, np.array([1.0, 0, 0]), texts, k=2, exclude="a")
        assert [t for t, _ in ex] == ["text c", "text b"]     # closest last
        assert all(d != DEFAULT_DESIGN for _, d in ex)

    def test_prompt_takes_a_custom_rubric(self):
        from crc.agents.discipline.llm_review import build_prompt

        p = build_prompt("A paper.", [("Ex.", "Case Study")], system="RUBRIC")
        assert "RUBRIC" in p and "Case Study" in p
        assert p.endswith("<|im_start|>assistant\n")
