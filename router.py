import ollama
import json


def route_query(query: str) -> str:
    """
    Uses Llama 3.1 to classify a query as either 'sparse' (exact keywords) 
    or 'dense' (semantic meaning).
    """

    # We use a strict system prompt to force the LLM to act as a logic gate, not a chatbot
    prompt = f"""You are a database routing AI. Analyze the following user query.
    
    RULES:
    1. If the query contains specific clause numbers (e.g., "Section 4"), exact document IDs, or highly specific proper nouns, you must choose 'sparse'.
    2. If the query asks for policy explanations, general concepts, or situational advice (e.g., "Am I allowed to..."), you must choose 'dense'.
    
    Output strictly in JSON format with a single key "track" and value either "sparse" or "dense".
    
    Query: {query}
    """

    print(f"Thinking about: '{query}'...")

    # We force Ollama to output valid JSON by passing format='json'
    response = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': prompt}],
        format='json'
    )

    # Parse the JSON string into a Python dictionary
    decision = json.loads(response['message']['content'])
    return decision['track']


# --- Test the Router ---
if __name__ == "__main__":
    # test: conceptual question (Should route to Dense/ChromaDB)
    query_1 = "What should I do if a vendor offers me a free dinner?"
    track_1 = route_query(query_1)
    print(f"Decision 1: Route to -> {track_1.upper()} TRACK\n")

    # Test 2: An exact keyword question (Should route to Sparse/BM25)
    query_2 = "What does Section 2.1 say?"
    track_2 = route_query(query_2)
    print(f"Decision 2: Route to -> {track_2.upper()} TRACK\n")
