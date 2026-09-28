"""Tests for the Agent 1 completion work: hierarchy maths, the retrieval
classifiers, the classical-baseline column contract, soft targets, the local-LLM
prompt/blend helpers, and the tuned fallback thresholds.

Everything here runs without a GPU, a checkpoint or llama-cpp: the model-bearing
paths are exercised through their pure functions.
"""
from __future__ import annotations

import numpy as np
import pytest

from crc.agents.discipline.hierarchy import (
    conditioned_field,
    field_to_discipline_marginal,
    hierarchical_scores,
    joint_discipline_probs,
)
from crc.agents.discipline.retrieval import (
    PrincipalSubspaceClassifier,
    fit_temperature,
    knn_probs,
    topk_neighbours,
)
from crc.taxonomy import DISCIPLINES, label_paper
from crc.taxonomy.fields import DISCIPLINE_FIELD_IDS, GLOBAL_ID2LABEL

N_FIELDS = len(GLOBAL_ID2LABEL)


# ------------------------------------------------------------------ hierarchy
class TestMarginal:
    def test_rows_sum_to_one_and_follow_the_field_map(self):
        rng = np.random.default_rng(0)
        fp = rng.dirichlet(np.ones(N_FIELDS), size=5)
        m = field_to_discipline_marginal(fp)
        assert m.shape == (5, len(DISCIPLINES))
        np.testing.assert_allclose(m.sum(axis=1), 1.0)
        for j, d in enumerate(DISCIPLINES):
            np.testing.assert_allclose(m[:, j], fp[:, DISCIPLINE_FIELD_IDS[d]].sum(axis=1))

    def test_all_mass_on_one_field_names_its_discipline(self):
        for j, d in enumerate(DISCIPLINES):
            fp = np.zeros((1, N_FIELDS))
            fp[0, DISCIPLINE_FIELD_IDS[d][0]] = 1.0
            assert field_to_discipline_marginal(fp).argmax() == j


class TestJointDecoding:
    a = np.array([[0.6, 0.3, 0.05, 0.02, 0.02, 0.01]])
    b = np.array([[0.2, 0.7, 0.04, 0.02, 0.02, 0.02]])

    def test_weight_zero_is_agent1_unchanged(self):
        np.testing.assert_allclose(joint_discipline_probs(self.a, self.b, 0.0), self.a)

    def test_weight_one_is_the_marginal(self):
        np.testing.assert_allclose(joint_discipline_probs(self.a, self.b, 1.0),
                                   self.b, atol=1e-6)

    def test_output_is_a_distribution(self):
        p = joint_discipline_probs(self.a, self.b, 0.4)
        np.testing.assert_allclose(p.sum(), 1.0)
        assert (p >= 0).all()

    def test_geometric_pool_lets_either_expert_veto(self):
        # An expert that gives a class ~0 mass keeps it near 0 even if the other
        # is confident -- the property an arithmetic mean does not have.
        a = np.array([[0.98, 0.02, 0, 0, 0, 0]]) + 1e-12
        b = np.array([[1e-6, 0.999, 0, 0, 0, 0]]) + 1e-12
        assert joint_discipline_probs(a, b, 0.5).argmax() == 1

    def test_rejects_weight_outside_unit_interval(self):
        with pytest.raises(ValueError):
            joint_discipline_probs(self.a, self.b, 1.5)


class TestHierarchicalScores:
    def test_half_credit_for_right_discipline_wrong_field(self):
        cs = DISCIPLINES[0]
        f0, f1 = DISCIPLINE_FIELD_IDS[cs][:2]
        r = hierarchical_scores([cs, cs], [cs, cs], [f0, f0], [f0, f1])
        assert r["hierarchical_f1"] == pytest.approx(0.75)   # (2 + 1) / 4
        assert r["partial_credit_share"] == pytest.approx(0.5)

    def test_wrong_discipline_earns_nothing(self):
        d0, d1 = DISCIPLINES[0], DISCIPLINES[1]
        r = hierarchical_scores([d0], [d1], [DISCIPLINE_FIELD_IDS[d0][0]],
                                [DISCIPLINE_FIELD_IDS[d1][0]])
        assert r["hierarchical_f1"] == 0.0

    def test_conditioned_field_stays_on_the_ballot(self):
        logits = np.zeros((1, N_FIELDS))
        best_anywhere = DISCIPLINE_FIELD_IDS[DISCIPLINES[1]][0]
        logits[0, best_anywhere] = 10.0                     # outside CS's ballot
        cs_field = DISCIPLINE_FIELD_IDS[DISCIPLINES[0]][3]
        logits[0, cs_field] = 1.0
        assert conditioned_field(logits, [DISCIPLINES[0]])[0] == cs_field


# ------------------------------------------------------------------ retrieval
def _unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


class TestRetrieval:
    def test_topk_matches_brute_force(self):
        rng = np.random.default_rng(1)
        store, q = _unit(rng.normal(size=(200, 16))), _unit(rng.normal(size=(7, 16)))
        idx, sim = topk_neighbours(q, store, k=5, chunk=3)
        brute = np.argsort(-(q @ store.T), axis=1)[:, :5]
        np.testing.assert_array_equal(idx, brute)
        assert (np.diff(sim, axis=1) <= 1e-7).all()          # sorted descending

    def test_knn_votes_follow_neighbour_labels(self):
        idx = np.array([[0, 1, 2]])
        sim = np.array([[0.9, 0.8, 0.1]], dtype=np.float32)
        labels = np.array([4, 4, 0])
        p = knn_probs(idx, sim, labels, k=3, tau=0.05)
        assert p.argmax() == 4
        np.testing.assert_allclose(p.sum(), 1.0)

    def test_principal_subspace_separates_orthogonal_classes(self):
        rng = np.random.default_rng(2)
        dim, n = 24, 60
        X, y = [], []
        for c in range(len(DISCIPLINES)):
            base = np.zeros(dim)
            base[c * 4:(c + 1) * 4] = 1.0
            X.append(base + 0.1 * rng.normal(size=(n, dim)))
            y += [c] * n
        X, y = _unit(np.vstack(X)), np.array(y)
        clf = PrincipalSubspaceClassifier(r=2).fit(X, y)
        assert (clf.scores(X).argmax(1) == y).mean() > 0.95

    def test_temperature_fit_prefers_sharper_when_scores_are_right(self):
        y = np.array([0, 1, 2])
        scores = np.eye(len(DISCIPLINES))[:3]
        assert fit_temperature(scores, y) < 1.0


# ------------------------------------------------------------------ classical baselines
def test_canonical_column_order_is_restored():
    from crc.agents.discipline.train_baseline import to_canonical

    classes = [3, 0, 5, 1, 4, 2]                  # a classifier's own order
    proba = np.eye(6)[[0]]                        # all mass on classes[0] == 3
    out = to_canonical(proba, classes)
    assert out.argmax() == 3


# ------------------------------------------------------------------ soft targets
class TestSoftTargets:
    def test_vote_distribution_matches_the_labeller(self):
        from crc.agents.discipline.train_soft import vote_distribution

        cats = ["cs.SD", "eess.AS", "cs.CL"]
        v = vote_distribution(cats)
        np.testing.assert_allclose(v.sum(), 1.0)
        assert DISCIPLINES[int(v.argmax())] == label_paper(cats).discipline

    def test_smoothing_keeps_a_distribution(self):
        from crc.agents.discipline.train_soft import smooth

        t = smooth(np.eye(len(DISCIPLINES))[[2]], 0.05)
        np.testing.assert_allclose(t.sum(), 1.0)
        assert t[0, 2] == pytest.approx(0.95 + 0.05 / len(DISCIPLINES))


# ------------------------------------------------------------------ local LLM helpers
class TestLLMHelpers:
    def test_prompt_is_chatml_and_ends_at_the_reply(self):
        from crc.agents.discipline.llm_review import build_prompt

        p = build_prompt("A paper.", [("Example one.", "Data Science")])
        assert p.startswith("<|im_start|>system\n")
        assert p.endswith("<|im_start|>assistant\n")
        # the example's answer is a completed assistant turn before the query
        assert "<|im_start|>assistant\nData Science<|im_end|>" in p
        assert p.index("Example one.") < p.index("A paper.")
        for d in DISCIPLINES:                     # the rubric names every label
            assert d in p

    def test_blend_is_a_normalised_convex_combination(self):
        from crc.agents.discipline.llm_review import LLMReview, blend_with_classifier

        clf = {d: 1.0 / len(DISCIPLINES) for d in DISCIPLINES}
        rv = LLMReview(label=DISCIPLINES[0], confidence=1.0,
                       probs={d: float(i == 0) for i, d in enumerate(DISCIPLINES)},
                       scores={}, model="stub")
        out = blend_with_classifier(clf, rv, llm_weight=0.5)
        assert sum(out.values()) == pytest.approx(1.0)
        assert max(out, key=out.get) == DISCIPLINES[0]
        assert blend_with_classifier(clf, rv, 0.0) == pytest.approx(clf)

    def test_model_lookup_is_case_insensitive(self, tmp_path, monkeypatch):
        from crc.agents.discipline import llm_review

        (tmp_path / "Qwen2.5-3B-Instruct-Q4_K_M.GGUF").write_bytes(b"")
        monkeypatch.setattr(llm_review, "MODEL_DIRS", [tmp_path])
        assert llm_review._resolve_model(None).name.lower() == \
            "qwen2.5-3b-instruct-q4_k_m.gguf"

    def test_missing_model_resolves_to_none(self, tmp_path, monkeypatch):
        from crc.agents.discipline import llm_review

        monkeypatch.setattr(llm_review, "MODEL_DIRS", [tmp_path / "absent"])
        assert llm_review._resolve_model(None) is None


# ------------------------------------------------------------------ thresholds
def test_tuned_fallback_thresholds_are_the_defaults():
    from crc.agents.discipline.predict import (
        LEGACY_LOW_CONFIDENCE,
        LEGACY_TIGHT_GAP,
        LOW_CONFIDENCE,
        TIGHT_GAP,
        borderline_check,
    )

    assert (LOW_CONFIDENCE, TIGHT_GAP) == (0.80, 0.04)
    assert (LEGACY_LOW_CONFIDENCE, LEGACY_TIGHT_GAP) == (0.55, 0.12)
    # 0.7 top-class confidence: flagged under the tuned pair, not the legacy one.
    assert borderline_check(0.70, 0.50)[0] is True
    assert borderline_check(0.70, 0.50, LEGACY_LOW_CONFIDENCE, LEGACY_TIGHT_GAP)[0] is False


# ------------------------------------------------------------------ TF-IDF member
class _FakeVec:
    def transform(self, texts):
        return np.zeros((len(texts), 1))


class _FakeClf:
    classes_ = np.array([5, 4, 3, 2, 1, 0])        # reversed order on purpose

    def predict_proba(self, X):
        p = np.array([0.05, 0.05, 0.05, 0.05, 0.2, 0.6])   # most mass on class 0
        return np.tile(p, (X.shape[0], 1))


class TestTfidfMember:
    def _member(self, t=1.0):
        from crc.agents.discipline.predict import TfidfMember
        return TfidfMember({"vectorizer": _FakeVec(), "selector": None,
                            "classifier": _FakeClf()}, temperature=t)

    def test_columns_follow_discipline_order(self):
        p = self._member().chunk_probs(["a", "b"])
        assert p.shape == (2, len(DISCIPLINES))
        assert p[0].argmax() == 0
        np.testing.assert_allclose(p.sum(axis=1), 1.0)

    def test_temperature_below_one_sharpens(self):
        raw = self._member(1.0).chunk_probs(["a"])
        sharp = self._member(0.8).chunk_probs(["a"])
        assert sharp.max() > raw.max()
        np.testing.assert_allclose(sharp.sum(), 1.0)

    def test_empty_input(self):
        assert self._member().chunk_probs([]).shape == (0, len(DISCIPLINES))


# ------------------------------------------------------------------ pipeline stages
class _StubDisc:
    name = "stub"

    def classify_document(self, doc):
        from crc.agents.discipline.predict import ClassificationResult
        probs = {d: 0.0 for d in DISCIPLINES}
        probs.update({"Computer Science": 0.55, "Data Science": 0.40,
                      "Information Systems": 0.05})
        return ClassificationResult(
            label="Computer Science", confidence=0.55, runner_up="Data Science",
            gap=0.15, probs=probs, doc_type="paragraph", n_chunks=1,
            strategy="weighted_mean", borderline=True, borderline_reason="stub",
            prediction_set=["Computer Science", "Data Science"], set_size=2)


class _StubField:
    name = "stub-field"

    def __init__(self):
        self.calls = []

    def predict_many(self, doc, disciplines):
        from crc.agents.field.predict import FieldPrediction
        self.calls.append(list(disciplines))
        return [FieldPrediction(discipline=d, label=f"{d} field", confidence=0.7,
                                runner_up="x", gap=0.3, probs={}, n_chunks=1,
                                borderline=False) for d in disciplines]


class TestPipelineStages:
    def test_run_is_built_from_stages(self):
        from crc.pipeline import Pipeline

        field = _StubField()
        pipe = Pipeline(_StubDisc(), field)
        st = pipe.stages(pipe_doc())
        res = pipe.run(pipe_doc())
        # the same candidates, in the same order, drive both views
        assert st["candidates"] == ["Computer Science", "Data Science"]
        assert res.discipline == "Computer Science"
        assert res.field == "Computer Science field"
        assert [a["discipline"] for a in res.alternatives] == ["Data Science"]
        assert res.discipline_set == ["Computer Science", "Data Science"]
        assert any("contested" in n for n in res.notes)
        # one encoder pass per document, however many candidates
        assert all(len(c) == 2 for c in field.calls)

    def test_without_agent2_the_field_is_skipped_not_invented(self):
        from crc.pipeline import Pipeline

        res = Pipeline(_StubDisc(), None).run(pipe_doc())
        assert res.field is None
        assert any("Agent 2 not loaded" in n for n in res.notes)


def pipe_doc():
    from crc.ingest import ingest
    return ingest("We propose a new algorithm for sorting and prove its complexity.")


# ------------------------------------------------------------------ calibration guard
class TestTemperatureGuard:
    def test_needs_to_help_on_validation(self):
        from crc.agents.discipline.calibrate import adopt_temperature
        assert adopt_temperature(0.03, 0.04) is False
        assert adopt_temperature(0.04, 0.03) is True

    def test_rejected_when_it_hurts_the_matched_slice(self):
        # the scibert-soft case: better on val (no ambiguous papers), worse on
        # the 2025+ half whose composition matches real input
        from crc.agents.discipline.calibrate import adopt_temperature
        assert adopt_temperature(0.0344, 0.0212, 0.0435, 0.0887) is False
        assert adopt_temperature(0.0344, 0.0212, 0.0435, 0.0300) is True


def test_deployed_model_is_one_named_directory():
    from crc.agents.discipline import DEPLOYED_MODEL
    from crc.pipeline import MODELS
    assert (MODELS / DEPLOYED_MODEL / "config.json").exists()
    assert (MODELS / DEPLOYED_MODEL / "conformal.json").exists()
