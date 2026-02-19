# Customer Support Ticket Autocomplete RAG System (Refactored)
# Purpose: Suggest support replies using past tickets

import ollama
from dataclasses import dataclass
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Ticket:
    key: str
    content: str
    file_origin: str = "Support_Tickets.txt"


def read_ticket_archive(path):
    """Read past support tickets from file"""
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    entries = [e.strip() for e in raw_text.split("\n\n") if len(e.strip()) > 20]

    ticket_list = []
    for i, entry in enumerate(entries):
        ticket_list.append(Ticket(key=f"case_{i+1}", content=entry))

    return ticket_list


class TicketRetriever:
    def __init__(self, tickets: List[Ticket]):
        self.tickets = tickets
        self.vector_model = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.ticket_vectors = self.vector_model.fit_transform([t.content for t in tickets])

    def find_matches(self, issue_text, limit=5):
        """Retrieve most relevant past tickets"""
        issue_vec = self.vector_model.transform([issue_text])
        similarity = cosine_similarity(issue_vec, self.ticket_vectors).flatten()

        best_indices = np.argsort(similarity)[::-1][:limit]

        matches = []
        for i in best_indices:
            if similarity[i] > 0.04:
                matches.append((self.tickets[i], float(similarity[i])))

        return matches


def generate_reply(context_block, new_issue):
    """Use Phi-3 to draft support response"""
    message = f"""
You are a Customer Support Assistant helping agents draft responses.
Your goal is to create a polite, professional response to a customer issue based on how similar past tickets were resolved.
Maintain a friendly, helpful tone.

Similar Past Tickets:
{context_block}

New Customer Issue:
{new_issue}

Suggested Response:"""

    result = ollama.chat(
        model="phi3:latest",
        messages=[{"role": "user", "content": message}]
    )

    return result["message"]["content"]


def start_system():
    print("=== Customer Support Ticket Autocomplete ===")
    print("Loading historical ticket database...\n")

    try:
        tickets = read_ticket_archive("Support_Tickets.txt")
        print(f"✓ Loaded {len(tickets)} historical support tickets.")
    except FileNotFoundError:
        print("✗ Error: 'Support_Tickets.txt' not found in current directory.")
        return

    retriever = TicketRetriever(tickets)

    while True:
        issue = input("\nEnter New Ticket Issue (or type 'exit' to quit): ")

        if issue.lower() in ["exit", "quit"]:
            print("\nThank you for using Ticket Autocomplete!")
            break

        similar_cases = retriever.find_matches(issue)

        if not similar_cases:
            print("No similar past tickets found. Suggest manual investigation.")
            continue

        combined_context = "\n\n".join([t.content for t, _ in similar_cases])

        print(f"\n{'='*60}")
        print(f"Found {len(similar_cases)} Similar Past Tickets")
        print(f"{'='*60}")
        print(combined_context)

        suggestion = generate_reply(combined_context, issue)

        print(f"\n{'='*60}")
        print("AI-Generated Response Suggestion")
        print(f"{'='*60}")
        print(suggestion)


if __name__ == "__main__":
    start_system()
