from pathlib import Path

from langchain_core.documents import Document
# pyrefly: ignore[missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
# pyrefly: ignore[missing-import]
from pypdf import PdfReader

from config import CHUNK_SIZE, CHUNK_OVERLAP

#This Python file is all about reading the pages from the PDF using PyPDF, making the page dictionary, chunking those, and making the long chain document list. 
def extract_pages(pdf_path: str) -> dict[int, str]:
    """Read a PDF and return {page_number: page_text} (page numbers start at 1)."""
    reader = PdfReader(pdf_path)

    if reader.is_encrypted:
        raise ValueError("This PDF is password-protected and can't be read.")

    pages: dict[int, str] = {}
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages[page_number] = text
    return pages


def chunk_pages(pages: dict[int, str], source_name: str) -> list[Document]:
    """Split each page into overlapping chunks and attach metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks: list[Document] = []
    for page_number, page_text in pages.items():
        for piece in splitter.split_text(page_text):
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "source": source_name,
                        "page": page_number,
                        "chunk_id": len(chunks),
                    },
                )
            )
    return chunks


def process_pdf(pdf_path: str) -> list[Document]:
    """Full pipeline for one PDF: extract → chunk."""
    pages = extract_pages(pdf_path)
    if not pages:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned/image-only PDF, which needs OCR (not part of V1)."
        )
    return chunk_pages(pages, source_name=Path(pdf_path).name)