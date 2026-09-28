"""Inputs the system must refuse or flag rather than classify.

Found by throwing awkward inputs at the full pipeline (2026-09-26): an empty
string, a scanned PDF, a mistyped path and a .pptx were all "classified" --
the path string itself was read as a paper -- and a one-word input came back at
81% confidence. Unreadable input now raises with a reason; short or non-English
input is answered with a warning shown first.
"""
from __future__ import annotations

import pytest

from crc.ingest import IngestError, ingest
from crc.pipeline import MIN_WORDS, input_warnings

ENGLISH = ("We present a tool that detects flaky tests in continuous integration by "
           "re-running failing tests under controlled perturbations, and we evaluate it "
           "on fifty open-source projects where it finds many flaky tests.")


class TestRefused:
    @pytest.mark.parametrize("text", ["", "   ", "\n\t  "])
    def test_no_text(self, text):
        with pytest.raises(IngestError, match="no text"):
            ingest(text)

    @pytest.mark.parametrize("path", ["C:/nowhere/paper.pdf", "paper.pdf",
                                      r"D:\papers\missing.docx"])
    def test_missing_file_is_not_read_as_text(self, path):
        with pytest.raises(FileNotFoundError):
            ingest(path)

    def test_unsupported_type(self, tmp_path):
        p = tmp_path / "slides.pptx"
        p.write_bytes(b"PK\x03\x04 not really")
        with pytest.raises(IngestError, match="Unsupported file type"):
            ingest(str(p))

    def test_scanned_pdf_names_ocr(self, tmp_path):
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        page.draw_rect(fitz.Rect(50, 50, 300, 300), color=(0, 0, 0))  # a picture, no text
        p = tmp_path / "scan.pdf"
        doc.save(p)
        with pytest.raises(IngestError, match="OCR"):
            ingest(str(p))


class TestStillText:
    def test_sentence_ending_in_a_dotted_name(self):
        # has spaces and no path separator, so it is text, not a missing file
        doc = ingest("We port the scheduler to Node.js")
        assert "scheduler" in doc.text

    def test_existing_txt_file(self, tmp_path):
        p = tmp_path / "abstract.txt"
        p.write_text(ENGLISH, encoding="utf-8")
        assert "flaky" in ingest(str(p)).text


class TestWarnings:
    def test_short_input(self):
        w = input_warnings(ingest("Blockchain."))
        assert len(w) == 1 and "1 word)" in w[0] and str(MIN_WORDS) in w[0]

    def test_not_english(self):
        fr = ("Nous proposons une nouvelle méthode d'apprentissage profond pour la "
              "détection d'intrusions dans les réseaux. Les expériences sur trois jeux "
              "de données montrent une amélioration de la précision.")
        assert any("English" in w for w in input_warnings(ingest(fr)))

    def test_ordinary_abstract_has_none(self):
        assert input_warnings(ingest(ENGLISH)) == []


def test_cli_reports_unreadable_input_without_a_traceback(capsys):
    from crc.cli import main

    assert main(["inspect", "C:/nowhere/paper.pdf"]) == 2
    assert "No such file" in capsys.readouterr().err


class TestPdfTitle:
    """A title that wraps onto several lines must come back whole (every chunk
    carries it into Agents 1 and 2), without disturbing section detection."""

    def _pdf(self, tmp_path, title_lines, size=22):
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        y = 72
        for line in title_lines:
            page.insert_text((72, y), line, fontsize=size)
            y += size + 6
        body = ("We present a system and evaluate it on three datasets. " * 3).strip()
        for _ in range(12):
            page.insert_text((72, y), body[:90], fontsize=10)
            y += 14
        p = tmp_path / "t.pdf"
        doc.save(p)
        return str(p)

    def test_wrapped_title_is_joined(self, tmp_path):
        from crc.ingest.parsers import parse_pdf

        d = parse_pdf(self._pdf(tmp_path, ["PDF Knowledge Extraction System with RAG and",
                                           "CAG Models"]))
        assert d.title == "PDF Knowledge Extraction System with RAG and CAG Models"

    def test_hyphenated_break_is_rejoined(self, tmp_path):
        from crc.ingest.parsers import parse_pdf

        d = parse_pdf(self._pdf(tmp_path, ["Scalable Knowledge Extrac-", "tion from PDFs"]))
        assert d.title == "Scalable Knowledge Extraction from PDFs"

    def test_single_line_title_unchanged(self, tmp_path):
        from crc.ingest.parsers import parse_pdf

        d = parse_pdf(self._pdf(tmp_path, ["A Conceptual Model of Research Methodology"]))
        assert d.title == "A Conceptual Model of Research Methodology"
