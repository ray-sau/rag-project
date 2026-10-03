import ollama
from router import route_query
from search import retrieve_context


def generate_answer(query: str, context: str) -> str:
    system_prompt = f"""You are an automated text extraction algorithm. 
    Your sole function is to evaluate the provided CONTEXT and extract the facts relevant to the user's query.
    
    RULES:
    1. Base your answer STRICTLY on the CONTEXT. Do not offer opinions or advice.
    2. Map common synonyms to bridge the query and the text (e.g., treat 'brother' as 'relative', 'vendor' as 'supplier', and 'buy stock' as 'financial interest').
    3. If the CONTEXT does not contain the necessary facts, output exactly: "I cannot answer this based on the provided documents."
    4. Never output conversational filler or safety warnings. Just extract the facts.
    
    CONTEXT:
    {context}
    """

    response = ollama.chat(
        model='llama3.1',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query}
        ]
    )
    return response['message']['content']


if __name__ == "__main__":
    print("\nSystem Ready. Type 'exit' to quit.\n")

    while True:
        user_query = input("Ask a policy question: ")
        if user_query.lower() in ['exit', 'quit']:
            break

        print("\nRouting...")
        track = route_query(user_query)
        print(f"-> Directed to: {track.upper()} database")

        print("Retrieving context...")
        context = retrieve_context(user_query, track)

        print("Generating answer...")
        answer = generate_answer(user_query, context)

        print("\n================ FINAL ANSWER ================")
        print(answer)
        print("==============================================\n")
