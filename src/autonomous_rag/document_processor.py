import re
from pathlib import Path

from langchain_core.documents import Document
# pyrefly: ignore[missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
# pyrefly: ignore[missing-import]
from pypdf import PdfReader

from config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_UPLOAD_MB
from autonomous_rag.exceptions import DocumentProcessingError
from autonomous_rag.logger import get_logger

logger = get_logger("documents")


def clean_text(text: str) -> str:
    """Collapse the messy line breaks and repeated spaces from PDF extraction."""
    text = text.replace("\x00", "")
    text = re.sub(r"\s+", " ", text)   # every run of spaces/newlines becomes one space
    return text.strip()


def extract_pages(pdf_path: str) -> dict[int, str]:
    """Read a PDF and return {page_number: page_text} (page numbers start at 1)."""
    path = Path(pdf_path)

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise DocumentProcessingError(
            f"{path.name} is {size_mb:.1f} MB. The limit is {MAX_UPLOAD_MB} MB."
        )

    try:
        reader = PdfReader(pdf_path)
        if reader.is_encrypted:
            raise DocumentProcessingError(
                f"{path.name} is password-protected and can't be read."
            )
        total_pages = len(reader.pages)
    except DocumentProcessingError:
        raise
    except Exception as error:   # corrupt file, not a PDF, truncated download...
        logger.exception("Could not open %s", path.name)
        raise DocumentProcessingError(
            f"{path.name} doesn't look like a valid, readable PDF."
        ) from error

    pages: dict[int, str] = {}
    unreadable = 0
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception:
            # One bad page shouldn't throw away the whole document
            logger.warning("Could not read page %d of %s", page_number, path.name, exc_info=True)
            unreadable += 1
            continue
        if text:
            pages[page_number] = text

    logger.info(
        "Extracted %s | pages=%d with_text=%d unreadable=%d",
        path.name, total_pages, len(pages), unreadable,
    )
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
        raise DocumentProcessingError(
            f"No text could be extracted from {Path(pdf_path).name}. "
            "It may be a scanned (image-only) PDF, which needs OCR. OCR isn't supported yet."
        )
    return chunk_pages(pages, source_name=Path(pdf_path).name)