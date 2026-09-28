"""Regression tests for ingestion, chunking and the document-type router.

Every case here corresponds to a bug that was found by running against real
papers, not to a hypothetical. The comments name the failure each one guards.
"""
from __future__ import annotations

import pytest

from crc.ingest import DocType, chunk_document, detect, parse_markdown, parse_text
from crc.ingest.chunkers import (
    ChunkConfig,
    chunk_article,
    chunk_paper,
    chunk_paragraph,
    sections_from_markdown,
)
from crc.ingest.doctype import candidate_headings
from crc.ingest.schema import Section, is_strict_heading_name, normalise_heading


class TestNormaliseHeading:
    @pytest.mark.parametrize("raw,expected", [
        ("Abstract", Section.ABSTRACT),
        ("1 Introduction", Section.INTRODUCTION),
        ("I. INTRODUCTION", Section.INTRODUCTION),
        ("## 3 Methodology", Section.METHODS),
        ("Experiments", Section.RESULTS),
        ("Conclusion", Section.CONCLUSION),
        ("References", Section.REFERENCES),
    ])
    def test_canonical(self, raw, expected):
        assert normalise_heading(raw) is expected

    def test_plural_related_works(self):
        # `work\b` never matched "Works"; every "Related Works" section in the
        # corpus was falling through to OTHER.
        assert normalise_heading("2 Related Works") is Section.RELATED_WORK
        assert normalise_heading("Related Work") is Section.RELATED_WORK

    def test_markdown_italic_stripped(self):
        # LaTeX-derived markdown writes italic headings as _Dataset_; leaving
        # the underscores attached defeated every pattern.
        assert normalise_heading("_Dataset_") is Section.METHODS
        assert normalise_heading("_Metrics_") is Section.METHODS

    def test_content_named_sections(self):
        for raw in ("Datasets", "Baselines", "Experimental Settings",
                    "Training Details", "Loss Function"):
            assert normalise_heading(raw) is Section.METHODS, raw
        assert normalise_heading("Ablations") is Section.RESULTS

    def test_back_matter_not_read_as_method(self):
        # "Data Availability" must not be read as a dataset section, and
        # "Author Contributions" must not be read as a method.
        assert normalise_heading("Data Availability") is Section.ACKNOWLEDGEMENTS
        assert normalise_heading("Author Contributions") is Section.ACKNOWLEDGEMENTS
        assert normalise_heading("Conflict of Interest") is Section.ACKNOWLEDGEMENTS


class TestStrictHeadingName:
    """The permissive/strict split — the single most important distinction."""

    def test_accepts_real_headings(self):
        for raw in ("Introduction", "3. Methodology", "RELATED WORKS",
                    "Results", "Conclusion", "Dataset"):
            assert is_strict_heading_name(raw), raw

    def test_rejects_prose_containing_section_words(self):
        # These sliced papers at nonsense points before the strict test existed.
        for raw in ("early stopping. We propose a mutual learning frame",
                    "fine-tune the models for domain-level and fine-gra",
                    "We evaluate our approach on three datasets",
                    "The results show that our model outperforms"):
            assert not is_strict_heading_name(raw), raw

    def test_permissive_and_strict_disagree_by_design(self):
        prose = "we train the models on a large corpus"
        assert normalise_heading(prose) is Section.METHODS   # permissive: fine
        assert not is_strict_heading_name(prose)             # strict: rejects


class TestCandidateHeadings:
    def test_rejects_float_captions(self):
        text = "TABLE III\nsome content here\nFIGURE 2\nmore content\n"
        assert not [h for h in candidate_headings(text)
                    if h.upper().startswith(("TABLE", "FIGURE"))]

    def test_rejects_bare_numbers(self):
        # Affiliation superscripts and page numbers.
        text = "1\nSchool of Computing\n2\nDepartment of Maths\n"
        assert not [h for h in candidate_headings(text) if h.strip().isdigit()]

    def test_accepts_numbered_and_allcaps(self):
        text = "1. INTRODUCTION\nbody text\nII. RELATED WORK\nmore body\n"
        heads = candidate_headings(text)
        assert any("INTRODUCTION" in h for h in heads)
        assert any("RELATED WORK" in h for h in heads)


class TestDocTypeRouter:
    def test_short_text_is_paragraph(self):
        v = detect("We propose a new method for image completion. " * 5)
        assert v.doc_type is DocType.PARAGRAPH

    def test_long_unstructured_is_article(self):
        v = detect("Machine learning is widely deployed. " * 200)
        assert v.doc_type is DocType.ARTICLE

    def test_structure_beats_length(self):
        # A short conference paper is still a paper. Testing word count first
        # collapsed these to one chunk and discarded the section labels.
        md = ("# T\n\n## Abstract\nx y z\n\n## 1 Introduction\na b c\n\n"
              "## 2 Methods\nd e f\n\n## 3 Results\ng h i\n\n"
              "## References\n[1] foo\n")
        _, sections = sections_from_markdown(md)
        v = detect(md, sections)
        assert v.doc_type is DocType.PAPER
        assert len(md.split()) < 350   # would be PARAGRAPH on length alone


class TestChunkers:
    def test_paragraph_is_single_chunk(self):
        chunks = chunk_paragraph("We propose a new method. " * 10)
        assert len(chunks) == 1
        assert chunks[0].section in (Section.ABSTRACT, Section.BODY)

    def test_paper_chunks_never_cross_sections(self):
        md = ("# Title\n\n## Abstract\n" + "alpha " * 60 +
              "\n\n## 1 Introduction\n" + "beta " * 60 +
              "\n\n## 2 Methods\n" + "gamma " * 60 + "\n")
        _, sections = sections_from_markdown(md)
        chunks = chunk_paper(sections, "Title")
        for c in chunks:
            words = set(c.text.split()) - {"Title"}
            # A chunk must draw from exactly one section's filler token.
            assert len(words & {"alpha", "beta", "gamma"}) <= 1, c.text[:80]

    def test_article_windows_overlap(self):
        paras = "\n\n".join(f"Paragraph {i} " + "word " * 80 for i in range(6))
        chunks = chunk_article(paras, cfg=ChunkConfig(target_words=100,
                                                      overlap_words=20))
        assert len(chunks) > 1
        assert all(c.section is Section.BODY for c in chunks)

    def test_max_chunks_respected(self):
        cfg = ChunkConfig(target_words=50, overlap_words=10, max_chunks=5)
        chunks = chunk_article("word " * 5000, cfg=cfg)
        assert len(chunks) <= 5

    def test_references_excluded_from_classification(self):
        md = ("# T\n\n## Abstract\n" + "alpha " * 40 +
              "\n\n## 1 Introduction\n" + "beta " * 40 +
              "\n\n## 2 Methods\n" + "gamma " * 40 +
              "\n\n## References\n[1] Smith 2020. [2] Jones 2021.\n")
        doc = chunk_document(parse_markdown(md))
        assert any(c.section is Section.REFERENCES for c in doc.chunks)
        assert all(c.section is not Section.REFERENCES
                   for c in doc.classifiable_chunks())


class TestInlineLeadIn:
    """IEEE/ACM run the abstract inline with its label — see parsers.py."""

    def test_inline_abstract_is_split_out(self):
        from crc.ingest.parsers import _tidy

        raw = ("Some Paper Title\nA. Author\nauthor@example.com\n"
               "Abstract-The given project includes a knowledge extraction "
               "system built on retrieval augmented generation.\n"
               "Keywords: retrieval, caching\n")
        out = _tidy(raw)
        # The label must end up on its own line so the slicer can find it.
        assert "\nAbstract\n" in out or out.startswith("Abstract\n")
        assert "\nKeywords\n" in out

    @pytest.mark.parametrize("sep", ["-", "—", "–", ":", "."])
    def test_all_separators(self, sep):
        from crc.ingest.parsers import _tidy

        out = _tidy(f"Abstract{sep}We propose a new method for search.")
        assert out.startswith("Abstract\n")

    def test_body_word_abstract_untouched(self):
        # "abstract" mid-sentence must not be split.
        from crc.ingest.parsers import _tidy

        out = _tidy("We use an abstract- syntax tree representation here.")
        assert "\n" not in out


class TestParseText:
    def test_pasted_abstract_round_trip(self):
        doc = parse_text("We introduce the on-line Viterbi algorithm for "
                         "decoding hidden Markov models in small space.")
        doc = chunk_document(doc)
        assert doc.doc_type is DocType.PARAGRAPH
        assert len(doc.chunks) == 1
