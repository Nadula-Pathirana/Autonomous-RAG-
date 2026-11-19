import pytest
# pyrefly: ignore[missing-import]
from pypdf import PdfWriter

from autonomous_rag.document_processor import chunk_pages, clean_text, process_pdf
from autonomous_rag.exceptions import DocumentProcessingError


def test_clean_text_collapses_whitespace():
    assert clean_text("Hello\n\n  world \t again\x00") == "Hello world again"


def test_chunks_keep_page_numbers_and_source():
    chunks = chunk_pages({1: "alpha " * 400, 3: "beta " * 50}, "doc.pdf")
    assert {c.metadata["page"] for c in chunks} == {1, 3}
    assert all(c.metadata["source"] == "doc.pdf" for c in chunks)
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))


def test_corrupt_file_raises_friendly_error(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"this is not a pdf")
    with pytest.raises(DocumentProcessingError):
        process_pdf(str(bad))


def test_pdf_without_text_raises_friendly_error(tmp_path):
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    blank = tmp_path / "blank.pdf"
    with open(blank, "wb") as f:
        writer.write(f)
    with pytest.raises(DocumentProcessingError):
        process_pdf(str(blank))