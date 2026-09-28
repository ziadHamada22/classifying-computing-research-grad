"""Tests for the field taxonomy (Agent 2) and the weak labeller."""
from __future__ import annotations

import numpy as np
import pytest

from crc.data.label_fields import label_one
from crc.taxonomy import DISCIPLINES
from crc.taxonomy.fields import (
    DISCIPLINE_FIELD_IDS,
    FIELDS,
    FIELDS_BY_DISCIPLINE,
    GLOBAL_LABEL2ID,
    N_GLOBAL_FIELDS,
    discipline_mask,
    label2id,
    n_fields,
)


class TestFieldTaxonomy:
    def test_every_discipline_has_fields(self):
        for d in DISCIPLINES:
            assert n_fields(d) >= 5, d

    def test_names_and_keys_unique(self):
        assert len({f.name for f in FIELDS}) == len(FIELDS)
        assert len({f.key for f in FIELDS}) == len(FIELDS)

    def test_every_field_has_evidence_and_prose(self):
        for f in FIELDS:
            assert f.description, f.name
            assert f.keywords, f"{f.name} has no keyword evidence"
            assert f.contrast, f"{f.name} has no boundary rule"

    def test_no_category_claimed_twice(self):
        seen: dict[str, str] = {}
        for f in FIELDS:
            for c in f.categories:
                assert c not in seen, f"{c} claimed by {seen.get(c)} and {f.name}"
                seen[c] = f.name

    def test_software_engineering_is_keyword_driven(self):
        # The whole reason this taxonomy exists: arXiv gives SE one category,
        # so its fields cannot depend on category evidence.
        se = FIELDS_BY_DISCIPLINE["Software Engineering"]
        assert len(se) == 7
        assert all(not f.categories for f in se), \
            "SE fields must not rely on arXiv categories"

    def test_field_discipline_backreference(self):
        for d in DISCIPLINES:
            for f in FIELDS_BY_DISCIPLINE[d]:
                assert f.discipline == d


class TestGlobalIndex:
    def test_global_index_covers_all_fields(self):
        assert N_GLOBAL_FIELDS == len(FIELDS)
        assert len(GLOBAL_LABEL2ID) == len(FIELDS)

    def test_discipline_ids_partition_the_space(self):
        seen: list[int] = []
        for d in DISCIPLINES:
            seen.extend(DISCIPLINE_FIELD_IDS[d])
        assert sorted(seen) == list(range(N_GLOBAL_FIELDS)), \
            "discipline field ids must partition the global space exactly"

    def test_mask_selects_only_that_discipline(self):
        for d in DISCIPLINES:
            m = discipline_mask(d)
            assert m.sum() == n_fields(d)
            assert set(np.flatnonzero(m)) == set(DISCIPLINE_FIELD_IDS[d])

    def test_local_index_is_contiguous(self):
        for d in DISCIPLINES:
            idx = label2id(d)
            assert sorted(idx.values()) == list(range(n_fields(d)))


class TestWeakLabeller:
    def _lab(self, disc, title, abstract="", cat=None):
        return label_one(disc, title, abstract, cat)

    def test_category_evidence_labels_confidently(self):
        r = self._lab("Computer Science", "A new detector", "", "cs.CV")
        assert r.field == "Computer Vision"
        assert r.evidence in ("category", "both")

    def test_keyword_evidence_carries_software_engineering(self):
        r = self._lab(
            "Software Engineering",
            "Flaky test detection in continuous integration",
            "We study flaky tests and propose a test suite analysis with "
            "improved code coverage.")
        assert r.field == "Testing, Verification and Validation"
        assert r.evidence == "keyword"

    def test_se_fields_are_separable(self):
        cases = {
            "Requirements and Specification":
                ("Eliciting requirements from stakeholders",
                 "We present a requirement elicitation method using user "
                 "stories and use case models."),
            "Maintenance and Evolution":
                ("Technical debt in legacy systems",
                 "We study refactoring and technical debt during software "
                 "maintenance and software evolution."),
            "Software Process and Management":
                ("Agile adoption and effort estimation",
                 "A study of scrum and devops practice with continuous "
                 "integration and effort estimation."),
        }
        for expected, (title, abstract) in cases.items():
            r = self._lab("Software Engineering", title, abstract)
            assert r.field == expected, f"{title!r} -> {r.field}"

    def test_insufficient_evidence_is_left_unlabelled(self):
        # A single incidental word must not decide the field.
        r = self._lab("Software Engineering", "A note on systems", "")
        assert r.field is None
        assert r.evidence == "none"

    def test_unknown_discipline_is_safe(self):
        r = self._lab("Not A Discipline", "anything", "anything")
        assert r.field is None

    def test_label_is_always_in_the_discipline(self):
        rng = np.random.default_rng(0)
        titles = ["deep learning for image segmentation",
                  "a database query optimizer",
                  "network protocol measurement",
                  "unit test generation with fuzzing",
                  "fpga accelerator design",
                  "bayesian hierarchical model"]
        for d in DISCIPLINES:
            for t in titles:
                r = label_one(d, t, t, None)
                if r.field is not None:
                    assert r.field in {f.name for f in FIELDS_BY_DISCIPLINE[d]}
