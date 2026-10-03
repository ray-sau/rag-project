import ollama
import bm25s
import chromadb
from fastembed import TextEmbedding

# ==========================================
# 1. INITIALIZE DATABASES & MODELS
# ==========================================
print("Loading databases and embedding models...")
chroma_client = chromadb.PersistentClient(path="./local_vector_db")
collection = chroma_client.get_collection(name="policy_collection")
bm25_retriever = bm25s.BM25.load("local_bm25_index", load_corpus=False)
embedding_model = TextEmbedding()

# ==========================================
# 2. HYBRID RETRIEVER (RRF)
# ==========================================


def retrieve_hybrid(query: str, k: int = 5) -> str:
    """
    Retrieves the top 'k' chunks from both BM25 and ChromaDB independently, 
    then merges them using Reciprocal Rank Fusion (RRF).
    """
    # Dense Retrieval (ChromaDB)
    query_embedding = list(embedding_model.embed([query]))[0].tolist()
    dense_results = collection.query(
        query_embeddings=[query_embedding], n_results=k)
    dense_chunks = dense_results['documents'][0]

    # Sparse Retrieval (BM25)
    query_tokens = bm25s.tokenize([query])
    indices, _ = bm25_retriever.retrieve(query_tokens, k=k)
    sparse_chunks = []
    for idx in indices[0]:
        res = collection.get(ids=[f"chunk_{idx}"])
        if res['documents']:
            sparse_chunks.append(res['documents'][0])

    # Reciprocal Rank Fusion (Merge & Score)
    chunk_scores = {}

    # Score Dense results
    for rank, chunk in enumerate(dense_chunks):
        if chunk not in chunk_scores:
            chunk_scores[chunk] = 0.0
        chunk_scores[chunk] += 1.0 / (rank + 1 + 60)  # RRF formula

    # Score Sparse results
    for rank, chunk in enumerate(sparse_chunks):
        if chunk not in chunk_scores:
            chunk_scores[chunk] = 0.0
        chunk_scores[chunk] += 1.0 / (rank + 1 + 60)  # RRF formula

    # Sort by the new fused score and take the absolute top 3 chunks
    sorted_chunks = sorted(chunk_scores.items(),
                           key=lambda item: item[1], reverse=True)
    best_chunks = [chunk for chunk, score in sorted_chunks[:3]]

    return "\n\n---\n\n".join(best_chunks)

# ==========================================
# 3. THE GENERATOR (Extraction Algorithm)
# ==========================================


def generate_answer(query: str, context: str) -> str:
    system_prompt = f"""You are an automated text extraction algorithm. 
    Your sole function is to evaluate the provided CONTEXT and extract the facts relevant to the user's query.
    
    RULES:
    1. Base your answer STRICTLY on the CONTEXT. Do not offer opinions or advice.
    2. Map common synonyms to bridge the query and the text (e.g., treat 'brother' as 'relative', and 'vendor' as 'supplier').
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


# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    print("\nSystem Ready. Type 'exit' to quit.\n")

    while True:
        user_query = input("Ask a policy question: ")
        if user_query.lower() in ['exit', 'quit']:
            break

        print("Retrieving context via Hybrid Search (RRF)...")
        document_context = retrieve_hybrid(user_query, k=5)

        print("Generating answer...")
        final_answer = generate_answer(user_query, document_context)

        print("\n================ FINAL ANSWER ================")
        print(final_answer)
        print("==============================================\n")
