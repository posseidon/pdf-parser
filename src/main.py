from typing import List, Dict, Any

from file.loader import PdfFileLoader, clean_text
from file.topic  import Topic
from sml.model import SmallLanguageModel
from vector.store import VectorStore
from datetime import datetime

def parse_pdf_file(filename: str) -> List[str]:
    """Parse the content of a file and return cleaned text."""
    loader = PdfFileLoader(filename)
    text = loader.extract_text().chunk_text()
    return text

def ask_question(question: str, store: VectorStore, llm: SmallLanguageModel, n_results: int = 3) -> Dict[str, Any]:
    """Answer a natural language question about the PDF"""
    try:
        # Search for relevant chunks
        results = store.search(question, n_results=n_results)
        
        if not results:
            return {"answer": "No relevant content found", "sources": []}
        
        # Combine top results as context
        context = " ".join([result["text"] for result in results[:n_results]])
        
        # Get answer from language model
        answer = llm.answer_question(question, context)
        
        return {
            "answer": answer,
            "sources": [
                {
                    "text": result["text"],
                    "score": result["score"]
                }
                for result in results
            ]
        }
        
    except Exception as e:
        return {"error": str(e)}

def q_and_a(text: List[str], filename: str) -> None:
    llm = SmallLanguageModel()
    
    vector_store = VectorStore()
    collection_name = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    vector_store.create_collection(collection_name)
    metadata = [{"chunk_id": i, "filename": filename} for i in range(len(text))]
    vector_store.add_documents(text, metadata)

    while True:
        question = input("\nAsk a question (or 'quit' to exit): ")
        
        if question.lower() == 'quit':
            break
        else:
            results = ask_question(question, vector_store, llm)
            if "error" not in results:
                print(f"\nAnswer: {results['answer']}")
            else:
                print(f"Error: {results['error']}")

def main():
    filename = "tests/resources/Hungary_short_history.pdf"
    text = parse_pdf_file(filename)
    tp = Topic()
    tp.fit(text)
    result = tp.get_topics()
    print(f"Identified topics: {result}")




if __name__ == "__main__":
    main()
