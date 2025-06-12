from typing import List, Dict, Any, Optional

import logging

from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmallLanguageModel:
    """Wrapper for small language models optimized for limited resources"""
    
    def __init__(self, model_path: str = "deepset/xlm-roberta-base-squad2"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.qa_pipeline = None
        self.load_model()
    
    def load_model(self):
        """Load the best available small Hungarian language model"""
        try:
            logger.info(f"Loading multilingual QA model from {self.model_path}")

            # Use a multilingual QA model that supports Hungarian (small and efficient)
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model_path,
                tokenizer=self.model_path,
                device=-1  # Use CPU
            )

            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            logger.info("Multilingual (including Hungarian) QA model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def answer_question(self, question: str, context: str) -> str:
        """Answer a question based on provided context"""
        try:
            if not self.qa_pipeline:
                return "Model not loaded"
            

            
            result = self.qa_pipeline(question=question, context=context)
            print(f"Question: {question}, Result: {result}")
            return result["answer"]
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return "Unable to answer question"
    
    def generate_quiz_questions(self, content: str, num_questions: int = 5) -> List[Dict]:
        """Generate quiz questions from content"""
        quiz_questions = []
        
        # Simple pattern-based question generation
        sentences = content.split('. ')
        
        for i, sentence in enumerate(sentences[:num_questions]):
            if len(sentence) > 20:  # Only use substantial sentences
                # Create a fill-in-the-blank question
                words = sentence.split()
                if len(words) > 5:
                    # Remove a key word (usually a noun or important term)
                    key_word_idx = len(words) // 2
                    key_word = words[key_word_idx]
                    question_text = sentence.replace(key_word, "______")
                    
                    quiz_questions.append({
                        "question": question_text,
                        "answer": key_word,
                        "type": "fill_blank",
                        "context": sentence
                    })
        
        return quiz_questions
    
    def generate_cue_cards(self, content: str, num_cards: int = 10) -> List[Dict]:
        """Generate cue cards from content"""
        cue_cards = []
        sentences = content.split('. ')
        
        for i, sentence in enumerate(sentences[:num_cards]):
            if len(sentence) > 30:  # Substantial content
                # Simple keyword extraction using TF-IDF
                words = sentence.lower().split()
                important_words = [w for w in words if len(w) > 4][:3]
                
                if important_words:
                    front = f"Mit említ: {', '.join(important_words)}?"
                    back = sentence.strip()
                    
                    cue_cards.append({
                        "front": front,
                        "back": back,
                        "keywords": important_words
                    })
        
        return cue_cards