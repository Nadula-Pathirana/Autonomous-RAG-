from autonomous_rag.sources import dedupe_sources, document_source, mark_cited, web_source


def test_dedupe_removes_repeats():
    sources = [
        document_source("a.pdf", 1, "text"),
        document_source("a.pdf", 1, "text"),
        document_source("a.pdf", 2, "text"),
    ]
    assert len(dedupe_sources(sources)) == 2


def test_document_is_cited_when_file_name_is_in_answer():
    sources = mark_cited([document_source("sample.pdf", 2, "x")], "See [sample.pdf, p.2].")
    assert sources[0]["cited"] is True


def test_document_is_not_cited_when_missing_from_answer():
    sources = mark_cited([document_source("sample.pdf", 2, "x")], "I don't know.")
    assert sources[0]["cited"] is False


def test_web_source_cited_by_domain():
    source = web_source("News", "https://groq.com/news", "x")
    assert mark_cited([source], "According to [News](https://groq.com/news)")[0]["cited"] is True
    assert mark_cited([source], "No sources here")[0]["cited"] is False


def test_score_is_rounded():
    assert document_source("a.pdf", 1, "x", 0.123456)["score"] == 0.123

def test_document_not_cited_when_page_differs():
    sources = mark_cited([document_source("sample.pdf", 3, "x")], "See [sample.pdf, p.2].")
    assert sources[0]["cited"] is False


def test_page_12_is_not_matched_by_page_1():
    sources = mark_cited([document_source("sample.pdf", 1, "x")], "See [sample.pdf, p.12].")
    assert sources[0]["cited"] is False