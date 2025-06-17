from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
import chromadb

class VectorStore:
    """Handles document embeddings and similarity search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Use a lightweight sentence transformer model
        self.embedding_model = SentenceTransformer(model_name)
        self.client = chromadb.Client()
        self.collection = None
    
    def create_collection(self, collection_name: str):
        """Create a new ChromaDB collection.

        Any existing collection with the same name is deleted first.
        """
        try:
            self.collection = self.client.delete_collection(collection_name)
        except:
            pass
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, chunks: List[str], metadata: List[Dict] = None):
        """Add document chunks to vector store"""
        if not self.collection:
            raise ValueError("Collection not created")
        
        embeddings = self.embedding_model.encode(chunks)
        
        if metadata is None:
            metadata = [{"chunk_id": i} for i in range(len(chunks))]
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadata,
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant document chunks"""
        if not self.collection:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        return [
            {
                "text": doc,
                "metadata": meta,
                "score": 1 - dist  # Convert distance to similarity
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]