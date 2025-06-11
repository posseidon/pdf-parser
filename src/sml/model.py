from typing import List, Dict, Any, Optional

import logging

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmallLanguageModel:
    """Wrapper for small language models optimized for limited resources"""
    
    def __init__(self, model_name: str = "SZTAKI-HLT/hunbert-base-cc"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.qa_pipeline = None
        self.load_model()
    
    def load_model(self):
        """Load the small Hungarian language model"""
        try:
            # For question answering, use a Hungarian QA model if available
            # Example: 'SZTAKI-HLT/hubertus-base-cc' or similar for Hungarian QA
            self.qa_pipeline = pipeline(
                "question-answering",
                model="mcsabai/huBert-fine-tuned-hungarian-squadv2",
                tokenizer="mcsabai/huBert-fine-tuned-hungarian-squadv2",
                device=-1  # Use CPU
            )
            
            # For text generation, use a Hungarian model if available
            self.tokenizer = AutoTokenizer.from_pretrained("mcsabai/huBert-fine-tuned-hungarian-squadv2")
            self.model = AutoModelForCausalLM.from_pretrained("mcsabai/huBert-fine-tuned-hungarian-squadv2")
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Hungarian models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Hungarian models: {e}")
    
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