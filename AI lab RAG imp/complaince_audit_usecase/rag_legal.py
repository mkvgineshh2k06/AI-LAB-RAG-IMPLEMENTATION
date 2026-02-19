# Legal Compliance RAG System (Refactored Version)
# Purpose: Audit service agreements and extract legal clauses

import ollama
from dataclasses import dataclass
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Clause:
    key: str
    body: str
    origin: str = "Service_Agreement.txt"


def read_contract(file_path):
    """Read contract file and divide into clauses"""
    with open(file_path, "r", encoding="utf-8") as f:
        text_data = f.read()

    parts = [p.strip() for p in text_data.split("\n\n") if len(p.strip()) > 20]

    clause_list = []
    for i, part in enumerate(parts):
        clause_list.append(Clause(key=f"section_{i+1}", body=part))

    return clause_list


class TfidfRetriever:
    def __init__(self, clause_list: List[Clause]):
        self.clauses = clause_list
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.tfidf.fit_transform([c.body for c in clause_list])

    def search(self, query_text, limit=5):
        """Find most relevant clauses"""
        query_vec = self.tfidf.transform([query_text])
        similarity_scores = cosine_similarity(query_vec, self.matrix).flatten()

        top_indices = np.argsort(similarity_scores)[::-1][:limit]

        results = []
        for i in top_indices:
            if similarity_scores[i] > 0.05:
                results.append((self.clauses[i], float(similarity_scores[i])))

        return results


def query_llm(context_text, user_query):
    """Send retrieved context to Phi-3 model"""
    instruction = f"""
You are a Legal Compliance Assistant specialized in contract analysis.
Extract the exact clause relevant to the user's question from the provided contract text.
Quote the relevant text directly and explain the legal implication based ONLY on the provided context.
Do not add information not present in the context.

Contract Context:
{context_text}

Compliance Question:
{user_query}

Answer:"""

    reply = ollama.chat(
        model="phi3:latest",
        messages=[{"role": "user", "content": instruction}]
    )

    return reply["message"]["content"]


def run_audit():
    print("=== Legal and Compliance Document Audit ===")
    print("Loading service agreement documents...\n")

    try:
        clauses = read_contract("Service_Agreement.txt")
        print(f"✓ Loaded {len(clauses)} legal clauses successfully.")
    except FileNotFoundError:
        print("✗ Error: 'Service_Agreement.txt' not found in current directory.")
        return

    retriever = TfidfRetriever(clauses)

    while True:
        question = input("\nAudit Question (type exit to quit): ")

        if question.lower() == "exit":
            break

        matches = retriever.search(question)

        if not matches:
            print("No relevant clauses found in the document.")
            continue

        combined_context = "\n\n".join([c.body for c, _ in matches])

        print(f"\n{'='*60}")
        print(f"Retrieved {len(matches)} Relevant Clauses")
        print(f"{'='*60}")
        print(combined_context)

        response = query_llm(combined_context, question)

        print(f"\n{'='*60}")
        print("Compliance Analysis")
        print(f"{'='*60}")
        print(response)


if __name__ == "__main__":
    run_audit()
