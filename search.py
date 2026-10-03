import bm25s
import chromadb
from fastembed import TextEmbedding

chroma_client = chromadb.PersistentClient(path="./local_vector_db")
collection = chroma_client.get_collection(name="policy_collection")
bm25_retriever = bm25s.BM25.load("local_bm25_index", load_corpus=False)
embedding_model = TextEmbedding()


def retrieve_dense(query: str, k: int = 2) -> list[str]:
    query_embedding = list(embedding_model.embed([query]))[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return results['documents'][0]


def retrieve_sparse(query: str, k: int = 2) -> list[str]:
    query_tokens = bm25s.tokenize([query])
    indices, _ = bm25_retriever.retrieve(query_tokens, k=k)
    retrieved_chunks = []
    for idx in indices[0]:
        res = collection.get(ids=[f"chunk_{idx}"])
        if res['documents']:
            retrieved_chunks.append(res['documents'][0])
    return retrieved_chunks


def retrieve_context(query: str, track: str, k: int = 2) -> str:
    if track == "sparse":
        chunks = retrieve_sparse(query, k=k)
    else:
        chunks = retrieve_dense(query, k=k)
    return "\n\n---\n\n".join(chunks)


if __name__ == "__main__":
    sample = "gifts from vendors"
    print("Testing Dense Retrieval:\n", retrieve_context(sample, "dense"))
