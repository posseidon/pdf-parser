"""Tests for the pdf parser utilities."""

import os
import sys
import types


# Ensure optional dependencies don't block import of the loader module.
sys.modules.setdefault("fitz", types.ModuleType("fitz"))
sys.modules.setdefault("PyPDF2", types.ModuleType("PyPDF2"))

# Add the ``src`` directory so ``file.loader`` can be imported when tests run
# from the project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from file.loader import clean_text


def test_clean_text_removes_control_chars_and_newlines():
    """``clean_text`` should strip control characters and collapse spaces."""

    sample = [
        "Hello\nWorld\x0c",
        "Goodbye\r\nWorld\x0b!",
    ]
    result = clean_text(sample)
    assert result == ["Hello World", "Goodbye World !"]
