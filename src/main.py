from file.loader import ParallelFileLoader, clean_text
from sml.model import SmallLanguageModel
from vector.store import VectorStore
from datetime import datetime

def parse_pdf_file(filename: str, loader: ParallelFileLoader) -> str:
    """Parse the content of a file and return cleaned text."""
    content = loader.extract_text()
    return clean_text(content)

def main():
    filename = "tests/resources/Hungary_short_history.pdf"

    llm = SmallLanguageModel()
    loader = ParallelFileLoader(filename)

    text = parse_pdf_file(filename, loader)
    chunks = loader.chunk_text(text)

    vector_store = VectorStore()
    collection_name = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    vector_store.create_collection(collection_name)
    metadata = [{"chunk_id": i, "filename": filename} for i in range(len(chunks))]
    vector_store.add_documents(chunks, metadata)

    results = vector_store.search("Mikor volt a honfoglalás?")
    print(results)
    
    # answer3 = llm.answer_question("Mikor volt a honfoglalás?", file_content)
    # answer4 = llm.answer_question("Kik alkották az első magyar kormányt?", file_content)
    # print(answer3)
    # print(answer4)

    

if __name__ == "__main__":
    main()
