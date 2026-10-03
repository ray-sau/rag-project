import json
import ollama


def route_query(query: str) -> str:
    prompt = f"""You are a database routing AI. Analyze the user query.
    RULES:
    1. If the query asks for a specific clause, section number, or exact name, output 'sparse'.
    2. If the query asks for concepts, rules, or situational advice, output 'dense'.
    
    Output strictly in JSON format with a single key "track" and value either "sparse" or "dense".
    
    Query: {query}
    """
    response = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': prompt}],
        format='json'
    )
    decision = json.loads(response['message']['content'])
    return decision.get('track', 'dense')


if __name__ == "__main__":
    test_query = "What should I do if a vendor offers me a free dinner?"
    print(f"Test Route: {route_query(test_query)}")
