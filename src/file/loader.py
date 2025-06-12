from typing import List, Dict, Any, Optional
import logging

import threading
import fitz  # PyMuPDF
import PyPDF2
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParallelFileLoader:
    def __init__(self, filename, chunk_size=1024):
        self.filename = filename
        self.chunk_size = chunk_size
        self.chunks = []
        self.lock = threading.Lock()

    def _read_chunk(self, start, size):
        try:
            with open(self.filename, 'rb') as f:
                f.seek(start)
                data = f.read(size)
                with self.lock:
                    self.chunks.append(data)
        except FileNotFoundError:
            print(f"File not found: {self.filename}")

    def load_in_chunks(self, num_threads=4):
        try:
            with open(self.filename, 'rb') as f:
                f.seek(0, 2)
                file_size = f.tell()
        except FileNotFoundError:
            print(f"File not found: {self.filename}")
            return []

        threads = []
        for i in range(0, file_size, self.chunk_size):
            t = threading.Thread(target=self._read_chunk, args=(i, self.chunk_size))
            threads.append(t)
            t.start()
            if len(threads) >= num_threads:
                for t in threads:
                    t.join()
                threads = []
        for t in threads:
            t.join()
        return self.chunks

    def extract_text_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF (better for complex layouts)"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting text with PyMuPDF: {e}")
            return ""
    
    def extract_text_pypdf2(self, pdf_path: str) -> str:
        """Fallback text extraction using PyPDF2"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting text with PyPDF2: {e}")
            return ""

    def extract_text(self) -> str:
        """Extract text from PDF using best available method"""
        text = self.extract_text_pymupdf(self.filename)
        if not text.strip():
            text = self.extract_text_pypdf2(self.filename)
        return text

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < self.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def parse_pdf_streams(self):
        """Parse PDF into text, images, tables, and links streams."""
        doc = fitz.open(self.filename)
        text_stream = []
        image_stream = []
        table_stream = []  # Table extraction is limited; see note below
        link_stream = []

        for page in doc:
            # Extract text
            text_stream.append(page.get_text())

            # Extract images
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_stream.append(base_image)

            # Extract links
            for link in page.get_links():
                link_stream.append(link)

            # Table extraction (PyMuPDF does not support table extraction directly)
            # For tables, consider using pdfplumber or camelot

        return {
            "text": text_stream,
            "images": image_stream,
            "tables": table_stream,
            "links": link_stream
        }

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E]+', '', text)
    text = text.strip()
    return text.encode('utf-8', errors='ignore').decode('utf-8')