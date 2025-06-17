import os
import pytest
from file.loader import PdfFileLoader, clean_text

@pytest.fixture
def sample_pdf(tmp_path):
    # Create a minimal PDF file for testing
    from PyPDF2 import PdfWriter
    pdf_path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return str(pdf_path)

def test_extract_text_empty_pdf(sample_pdf):
    loader = PdfFileLoader(sample_pdf)
    text = loader.extract_text()._text
    assert isinstance(text, str)
    assert text.strip() == ""  # Blank PDF should yield empty string

def test_chunk_text_no_text():
    loader = PdfFileLoader("dummy.pdf")
    loader._text = None
    with pytest.raises(ValueError):
        loader.chunk_text()

def test_chunk_text_by_sentence():
    loader = PdfFileLoader("dummy.pdf")
    loader._text = "Ez egy mondat. Ez a második! És ez a harmadik?"
    chunks = loader.chunk_text()
    assert chunks == [
        "Ez egy mondat.",
        "Ez a második!",
        "És ez a harmadik?"
    ]

def test_chunk_text_by_newline():
    loader = PdfFileLoader("dummy.pdf")
    loader._text = "Első bekezdés.\n\nMásodik bekezdés.\nHarmadik bekezdés."
    # If your chunk_text splits by sentence, adjust this test accordingly
    chunks = loader.chunk_text()
    assert "Első bekezdés." in chunks
    assert "Második bekezdés." in chunks or "Második bekezdés.\nHarmadik bekezdés." in chunks

def test_clean_text_preserves_utf8():
    text_list = ["áéűőúóüöÍÉÁŐŰ", "Hello\nWorld!"]
    cleaned = clean_text(text_list)
    assert any("áéűőúóüöÍÉÁŐŰ" in c for c in cleaned)
    assert all('\n' not in c for c in cleaned)

def test_extract_text_file_not_found():
    loader = PdfFileLoader("not_a_real_file.pdf")
    text = loader.extract_text()._text
    assert text == "" or text is None

def test_parse_pdf_streams_empty_pdf(sample_pdf):
    loader = PdfFileLoader(sample_pdf)
    streams = loader.parse_pdf_streams()
    assert isinstance(streams, dict)
    assert "text" in streams and "images" in streams and "links" in streams
