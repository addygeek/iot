"""Advanced Speech-Aware Summarizer"""

import os
import re
from datetime import datetime
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except:
    EMBEDDINGS_AVAILABLE = False


class Summarizer:
    def __init__(self, mode='detailed', num_sentences=5):
        self.mode = mode
        self.num_sentences = num_sentences
        
        self.embedding_model = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except:
                pass
        
        self.filler_words = {
            "okay", "uh", "um", "you know", "like", "so", "basically",
            "actually", "right", "hmm", "alright"
        }
    
    def generate_summary(self, text):
        if not text or len(text.strip()) < 20:
            return "Text too short to summarize."
        
        cleaned = self._clean_speech_text(text)
        
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(cleaned)
        except:
            sentences = cleaned.split('. ')
        
        if len(sentences) <= 3:
            return cleaned
        
        return self._semantic_summary(sentences, min(5, len(sentences)))
    
    def _clean_speech_text(self, text):
        lowered = text.lower()
        for w in self.filler_words:
            lowered = re.sub(rf"\b{w}\b", "", lowered)
        lowered = re.sub(r"\b(\w+)\s+\1\b", r"\1", lowered)
        return lowered.strip()
    
    def _semantic_summary(self, sentences, top_k=3):
        if not sentences:
            return ""
        
        if len(sentences) <= top_k:
            return " ".join(sentences)
        
        if EMBEDDINGS_AVAILABLE and self.embedding_model:
            embeddings = self.embedding_model.encode(sentences)
            sim_matrix = cosine_similarity(embeddings)
            scores = sim_matrix.sum(axis=1)
        else:
            scores = np.array([len(s.split()) for s in sentences])
        
        ranked = np.argsort(scores)[::-1][:top_k]
        ranked = sorted(ranked)
        selected = [sentences[i] for i in ranked]
        
        return "\n".join(selected)
    
    def save_summary(self, summary, session_folder, session_name):
        summary_file = os.path.join(session_folder, f"{session_name}_summary.txt")
        
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Summary: {session_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mode: {self.mode}\n")
            f.write("=" * 60 + "\n\n")
            f.write(summary)
            f.write("\n\n" + "=" * 60 + "\n")
        
        return summary_file
