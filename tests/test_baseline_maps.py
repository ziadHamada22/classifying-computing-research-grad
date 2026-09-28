"""Tests for the external-baseline label maps (OpenAlex, CSO)."""
from __future__ import annotations

import math

from crc.taxonomy import DISCIPLINES
from crc.taxonomy.cso_map import CSO_TOPIC_TO_DISCIPLINE, map_topics
from crc.taxonomy.openalex_map import (
    AMBIGUOUS_SUBFIELDS,
    SUBFIELD_TO_DISCIPLINE,
    is_ambiguous,
    map_subfield,
)


class TestOpenAlexMap:
    def test_all_targets_are_real_disciplines(self):
        for d in SUBFIELD_TO_DISCIPLINE.values():
            assert d in DISCIPLINES

    def test_known_subfields(self):
        assert map_subfield("Artificial Intelligence") == "Computer Science"
        assert map_subfield("Software") == "Software Engineering"
        assert map_subfield("Computer Networks and Communications") == "Information Technology"
        assert map_subfield("Hardware and Architecture") == "Computer Engineering"
        assert map_subfield("Statistics and Probability") == "Data Science"

    def test_robust_to_non_str(self):
        # Papers OpenAlex could not resolve arrive as NaN/None.
        assert map_subfield(None) is None
        assert map_subfield(float("nan")) is None
        assert map_subfield("Sociology and Political Science") is None  # unmapped
        assert is_ambiguous(None) is False

    def test_ambiguous_flagged(self):
        for s in AMBIGUOUS_SUBFIELDS:
            assert is_ambiguous(s)
            assert map_subfield(s) in DISCIPLINES  # still has a default


class TestCsoMap:
    def test_all_targets_are_real_disciplines(self):
        for d in CSO_TOPIC_TO_DISCIPLINE.values():
            assert d in DISCIPLINES

    def test_vote_picks_majority(self):
        topics = ["computer hardware", "signal processing", "sensors",  # 3x CE
                  "artificial intelligence"]                             # 1x CS
        assert map_topics(topics) == "Computer Engineering"

    def test_root_is_ignored(self):
        # "computer science" is the CSO root and must not vote.
        assert map_topics(["computer science"]) is None

    def test_tie_abstains(self):
        assert map_topics(["computer hardware", "software engineering"]) is None

    def test_empty_and_unmapped(self):
        assert map_topics([]) is None
        assert map_topics(["quantum widgets", "commerce"]) is None
