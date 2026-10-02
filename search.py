import bm25s
import chromadb
from fastembed import TextEmbedding

# Define the test query
# Try changing this to an exact phrase like "Section 4" to see BM25 win
query = "What is the policy on accepting gifts from vendors?"
print(f"\nQuery: '{query}'\n")

# Connect to ChromaDB (Our text storage and semantic engine)
print("Loading databases...")
chroma_client = chromadb.PersistentClient(path="./local_vector_db")
collection = chroma_client.get_collection(name="policy_collection")

# Query the Dense Track (ChromaDB + ONNX)
print("\n--- TRACK A: Semantic Search (ChromaDB) ---")
embedding_model = TextEmbedding()
query_embedding = list(embedding_model.embed([query]))[0].tolist()

chroma_results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

for i, text in enumerate(chroma_results['documents'][0]):
    print(f"Semantic Result {i+1}:\n{text}\n")


# Query the Sparse Track (BM25)
print("--- TRACK B: Exact Keyword Search (BM25) ---")

# Load the BM25 index
bm25_retriever = bm25s.BM25.load("local_bm25_index", load_corpus=False)

# Tokenize the user query and search
query_tokens = bm25s.tokenize([query])
bm25_indices, bm25_scores = bm25_retriever.retrieve(query_tokens, k=2)

# Safely fetch the exact text from ChromaDB using the unique chunk ID
for i, doc_index in enumerate(bm25_indices[0]):
    chunk_id = f"chunk_{doc_index}"
    doc_data = collection.get(ids=[chunk_id])

    # Ensure the document exists before printing to prevent crashes
    if doc_data['documents']:
        doc_text = doc_data['documents'][0]
        print(
            f"Keyword Result {i+1} (Score: {bm25_scores[0][i]:.2f}):\n{doc_text}\n")
