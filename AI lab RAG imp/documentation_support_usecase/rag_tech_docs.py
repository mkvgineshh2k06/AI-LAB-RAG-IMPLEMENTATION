# Technical Documentation RAG System (Refactored)
# Purpose: Search API docs and generate developer-friendly answers

import ollama
from dataclasses import dataclass
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ApiSection:
    uid: str
    content: str
    origin: str = "API_Docs_v2.txt"


def read_api_docs(file_path):
    """Load API documentation and split into sections"""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = f.read()

    sections = [s.strip() for s in raw_data.split("\n\n") if len(s.strip()) > 20]

    section_list = []
    for i, sec in enumerate(sections):
        section_list.append(ApiSection(uid=f"section_{i+1}", content=sec))

    return section_list


class ApiDocSearcher:
    def __init__(self, sections: List[ApiSection]):
        self.sections = sections
        self.model = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.section_vectors = self.model.fit_transform([s.content for s in sections])

    def search_docs(self, query_text, limit=4):
        """Retrieve most relevant documentation sections"""
        query_vector = self.model.transform([query_text])
        scores = cosine_similarity(query_vector, self.section_vectors).flatten()

        ranked_indices = np.argsort(scores)[::-1][:limit]

        matches = []
        for i in ranked_indices:
            if scores[i] > 0.02:
                matches.append((self.sections[i], float(scores[i])))

        return matches


def generate_technical_response(context_text, developer_query):
    """Use Phi-3 to produce technical explanation"""
    prompt_text = f"""
You are a Technical Documentation Assistant helping developers understand the API.
Provide clear, code-focused responses based on the documentation.
Include code snippets and examples when relevant.
Format code properly using markdown code blocks.

API Documentation:
{context_text}

Developer Question:
{developer_query}

Answer:"""

    output = ollama.chat(
        model="phi3:latest",
        messages=[{"role": "user", "content": prompt_text}]
    )

    return output["message"]["content"]


def launch_interface():
    print("=== Technical Documentation Support (API v2.0) ===")
    print("Loading API documentation...\n")

    try:
        api_sections = read_api_docs("API_Docs_v2.txt")
        print(f"✓ Loaded {len(api_sections)} documentation sections.")
    except FileNotFoundError:
        print("✗ Error: 'API_Docs_v2.txt' not found in current directory.")
        return

    search_engine = ApiDocSearcher(api_sections)

    while True:
        user_query = input("\nAsk Technical/API Question (or type 'exit' to quit): ")

        if user_query.lower() in ["exit", "quit"]:
            print("\nGoodbye!")
            break

        results = search_engine.search_docs(user_query)

        if not results:
            print("No matching documentation found.")
            continue

        combined_context = "\n\n".join([sec.content for sec, _ in results])

        print(f"\n[Retrieved {len(results)} relevant documentation sections]\n")
        print(combined_context)

        reply = generate_technical_response(combined_context, user_query)

        print("\n" + "="*60)
        print("Technical Answer")
        print("="*60)
        print(reply)


if __name__ == "__main__":
    launch_interface()
