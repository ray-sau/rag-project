import pdfplumber
import bm25s
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding

pdf_path = "gitlab_code_of_conduct.pdf"
text_corpus = []

# Extract the raw text
print("Extracting text from PDF...")
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            text_corpus.append(text)

full_text = "\n".join(text_corpus)

# Apply Recursive Character Splitting (Regex enabled)
print("Applying Recursive Character Splitting...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[r"\n{2,}", r"\n", r"\.", " ", ""],
    is_separator_regex=True
)
chunks = splitter.split_text(full_text)
print(f"Generated {len(chunks)} chunks.")

# Build and Save the Sparse (BM25) Track
print("\nBuilding BM25 exact-keyword index...")
corpus_tokens = bm25s.tokenize(chunks)
bm25_retriever = bm25s.BM25()
bm25_retriever.index(corpus_tokens)
bm25_retriever.save("local_bm25_index")
print("BM25 index saved safely to ./local_bm25_index")

# Build and Save the Dense (Chroma/ONNX) Track
print("\nGenerating ONNX embeddings and building ChromaDB...")
embedding_model = TextEmbedding()
all_embeddings = [e.tolist() for e in embedding_model.embed(chunks)]

chroma_client = chromadb.PersistentClient(path="./local_vector_db")

# Safely reset the collection by deleting it completely first
try:
    chroma_client.delete_collection(name="policy_collection")
except Exception:
    pass  # If it doesn't exist yet, just continue

collection = chroma_client.get_or_create_collection(name="policy_collection")

collection.add(
    documents=chunks,
    embeddings=all_embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
print("ChromaDB index saved safely to ./local_vector_db")
