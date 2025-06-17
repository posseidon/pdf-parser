import unittest
import sys
import types
import os

# Provide dummy modules for optional dependencies so the import works without
# requiring the heavy external packages.
sys.modules.setdefault("fitz", types.ModuleType("fitz"))
sys.modules.setdefault("PyPDF2", types.ModuleType("PyPDF2"))

# Ensure the repository root is on the Python path so `src` is importable
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from src.file.loader import clean_text

class TestCleanText(unittest.TestCase):
    def test_clean_text_basic(self):
        texts = ["Hello\nWorld\t!", "Good\rbye"]
        result = clean_text(texts)
        self.assertEqual(result, ["Hello World !", "Good bye"])

if __name__ == "__main__":
    unittest.main()
