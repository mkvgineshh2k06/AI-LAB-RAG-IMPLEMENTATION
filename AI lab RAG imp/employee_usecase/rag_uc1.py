# HR Policy Knowledge Base RAG System (Refactored)
# Purpose: Employee self-service queries for HR policies

import ollama
from dataclasses import dataclass
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class PolicySection:
    uid: str
    content: str
    file_source: str = "HR_Policy_2026.txt"


def read_policy_document(file_path):
    """Load HR policy file and divide into sections"""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    sections = [s.strip() for s in raw_text.split("\n\n") if len(s.strip()) > 20]

    policy_list = []
    for i, sec in enumerate(sections):
        policy_list.append(PolicySection(uid=f"section_{i+1}", content=sec))

    return policy_list


class PolicyRetriever:
    def __init__(self, sections: List[PolicySection]):
        self.sections = sections
        self.tfidf_model = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.section_vectors = self.tfidf_model.fit_transform([s.content for s in sections])

    def search_policies(self, question_text, limit=4):
        """Retrieve relevant policy sections"""
        q_vector = self.tfidf_model.transform([question_text])
        similarity_scores = cosine_similarity(q_vector, self.section_vectors).flatten()

        ranked_indices = np.argsort(similarity_scores)[::-1][:limit]

        matches = []
        for i in ranked_indices:
            if similarity_scores[i] > 0.03:
                matches.append((self.sections[i], float(similarity_scores[i])))

        return matches


def generate_hr_response(context_text, employee_query):
    """Use Phi-3 to produce HR policy answer"""
    instruction = f"""
You are an Internal HR Assistant helping employees understand company policies.
Answer the employee's question strictly based on the provided policy context.
Be clear, concise, and employee-friendly in your response.
If the answer is not in the context, say "I cannot find this information in the current HR policies."

HR Policy Context:
{context_text}

Employee Question:
{employee_query}

Answer:"""

    result = ollama.chat(
        model="phi3:latest",
        messages=[{"role": "user", "content": instruction}]
    )

    return result["message"]["content"]


def start_hr_portal():
    print("=== Internal Employee Knowledge Base (HR Policy 2026) ===")
    print("Initializing HR policy database...\n")

    try:
        policies = read_policy_document("HR_Policy_2026.txt")
        print(f"✓ Successfully loaded {len(policies)} policy sections.")
    except FileNotFoundError:
        print("✗ Error: 'HR_Policy_2026.txt' not found in current directory.")
        return

    retriever = PolicyRetriever(policies)

    while True:
        user_question = input("\nAsk HR Question (or type 'exit' to quit): ")

        if user_question.lower() in ["exit", "quit"]:
            print("\nThank you for using HR Knowledge Base!")
            break

        relevant_sections = retriever.search_policies(user_question)

        if not relevant_sections:
            print("No matching policy found.")
            continue

        combined_context = "\n\n".join([sec.content for sec, _ in relevant_sections])

        print(f"\n[Found {len(relevant_sections)} relevant policy sections]\n")

        reply = generate_hr_response(combined_context, user_question)

        print("\n" + "=" * 60)
        print("HR Policy Answer")
        print("=" * 60)
        print(reply)


if __name__ == "__main__":
    start_hr_portal()
