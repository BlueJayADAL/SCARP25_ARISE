import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class WorkoutTherapyFAQMatcher:
    def __init__(self, glove_path: str = None, faq_json_path: str = None, embedding_dim: int = 100):
        self.embedding_dim = embedding_dim
        self.glove_dict = {}
        self.faq_embeddings = {}
        self.faqs = {}
        self.latencies = []

        self.domain_stopwords = {'workout', 'exercise', 'therapy', 'physical', 'fitness'}

        self.synonym_map = {
    "arms": "biceps",
    "guns": "biceps",
    "abs": "core",
    "stomach": "core",
    "belly": "core",
    "quads": "quadriceps",
    "lats": "back",
    "pecs": "chest",
    "glute": "glutes",
    "butt": "glutes",
    "booty": "glutes",
    "delts": "deltoids",
    "shoulders": "deltoids",
    "tri": "triceps",
    "hams": "hamstrings",
    "calf": "calves",
    "leg": "legs",
    "legs": "legs",
    "upper body": "chest",
    "lower body": "legs",
    "midsection": "core",
    "sore": "pain",
    "ache": "pain",
    "hurt": "pain",
    "injured": "injury",
    "strain": "injury",
    "pull": "injury"
}  # Synonym map omitted here for brevity

        self.muscle_to_exercises = {
    "biceps": ["bicep curl", "hammer curl", "chin-up"],
    "triceps": ["tricep extension", "dips", "close-grip bench press"],
    "glutes": ["hip thrust", "glute bridge", "squat"],
    "hamstrings": ["romanian deadlift", "leg curl"],
    "quadriceps": ["squat", "leg press", "lunge"],
    "deltoids": ["shoulder press", "lateral raise", "arnold press"],
    "chest": ["bench press", "pushup", "incline press"],
    "core": ["plank", "sit-up", "leg raise"],
    "back": ["deadlift", "pull-up", "row"],
    "calves": ["calf raise", "jump rope"],
    "legs": ["squat", "lunge", "leg press"]
}  # Muscle to exercise map omitted here for brevity

        self.exercise_to_muscles = defaultdict(list)
        for muscle, exercises in self.muscle_to_exercises.items():
            for ex in exercises:
                self.exercise_to_muscles[ex].append(muscle)

        if glove_path:
            self.load_glove_embeddings(glove_path)
        if faq_json_path:
            self.load_faqs_from_json(faq_json_path)

    def load_glove_embeddings(self, glove_path: str):
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]])
                if len(vector) == self.embedding_dim:
                    self.glove_dict[word] = vector

    def preprocess_fitness_text(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s\-/]', '', text)
        words = text.split()
        return [self.synonym_map.get(w, w) for w in words if w not in self.domain_stopwords and len(w) > 2]

    def get_sentence_embedding(self, sentence: str) -> np.ndarray:
        words = self.preprocess_fitness_text(sentence)
        vectors = [self.glove_dict[w] for w in words if w in self.glove_dict]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.embedding_dim)

    def load_faqs_from_json(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            faq_list = json.load(f)
        self.faqs = {}
        self.faq_embeddings = {}
        for i, faq in enumerate(faq_list):
            self.faqs[i] = faq
            combined = faq['question'] + ' ' + faq['answer']
            self.faq_embeddings[i] = self.get_sentence_embedding(combined)

    def find_best_match(self, query: str, threshold: float = 0.4, top_k: int = 3, category_filter: Optional[List[str]] = None) -> List[Tuple[int, float, dict]]:
        if not self.faq_embeddings:
            return []
        query_emb = self.get_sentence_embedding(query)
        scores = []
        for i, emb in self.faq_embeddings.items():
            faq = self.faqs[i]
            if category_filter and faq.get("category") not in category_filter:
                continue
            score = cosine_similarity(query_emb.reshape(1, -1), emb.reshape(1, -1))[0][0]
            if score >= threshold:
                scores.append((i, score, faq))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    def handle_muscle_or_exercise_query(self, query: str) -> Optional[str]:
        words = self.preprocess_fitness_text(query)
        for word in words:
            if word in self.muscle_to_exercises:
                return f"To grow your {word}, try: {', '.join(self.muscle_to_exercises[word])}."
            if word in self.exercise_to_muscles:
                muscles = self.exercise_to_muscles[word]
                details = f"The exercise '{word}' primarily targets: {', '.join(muscles)}."
                if word in self.faqs:
                    faq_matches = [faq['answer'] for faq in self.faqs.values() if word in faq['question'].lower()]
                    if faq_matches:
                        details += f"\nAdditional info: {faq_matches[0]}"
                return details
        return None

    def respond_to_query(self, query: str, threshold: float = 0.4, top_k: int = 3) -> str:
        start = time.time()

        # Determine query category intent
        query_lower = query.lower()
        category_filter = None
        if any(kw in query_lower for kw in ["how to", "how do i", "form", "perform", "steps", "do a"]):
            category_filter = ["exercise"]
        elif any(kw in query_lower for kw in ["grow", "build", "bigger", "develop", "increase"]):
            category_filter = ["muscle_growth"]
        elif any(kw in query_lower for kw in ["what muscle", "what area", "target", "involve"]):
            category_filter = ["targeting", "exercise"]
        elif any(kw in query_lower for kw in ["injury", "pain", "hurt", "rehab", "recovery", "sore", "strain"]):
            category_filter = ["injury_recovery"]

        matches = self.find_best_match(query, threshold, top_k, category_filter)

        # Force logic to prioritize explicit exercise how-to over muscle mapping if keyword matches
        if category_filter == ["exercise"]:
            fallback = self.handle_muscle_or_exercise_query(query)
            if not matches and fallback:
                end = time.time()
                self.latencies.append(end - start)
                return fallback

        if matches:
            best = matches[0][2]
            response = f"{best['question']}\n{best['answer']}"
        else:
            response = self.handle_muscle_or_exercise_query(query)
            if not response:
                response = "I'm sorry, I couldn't find a good match for your question."

        end = time.time()
        self.latencies.append(end - start)
        return response

    def get_latency_summary(self) -> Dict[str, float]:
        if not self.latencies:
            return {"count": 0, "avg": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": len(self.latencies),
            "avg": sum(self.latencies) / len(self.latencies),
            "min": min(self.latencies),
            "max": max(self.latencies)
        }
