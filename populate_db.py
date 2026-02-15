import os
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from resume_text_extractor import load_documents

VECTOR_DB_PATH = "hf_vectorstore"


def build_vector_store(documents):
    """Build a brand new FAISS vector store from a list of documents"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(documents)

    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)

    print(f"Vector store built with {len(documents)} documents.")
    return vectorstore

def load_vector_store():
    """Load existing FAISS vector store if it exists"""
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

def add_new_documents(vectorstore, new_docs):
    """Add new resumes to existing vector DB without duplicating by checking filenames"""
    # Get existing filenames already in the vector store
    existing_sources = set(doc.metadata.get("source") for doc in vectorstore.docstore._dict.values())

    # Filter out docs already in DB
    docs_to_add = [doc for doc in new_docs if doc.metadata.get("source") not in existing_sources]

    if not docs_to_add:
        print("No new resumes to add. Vector store is up to date.")
        return vectorstore

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(docs_to_add)

    # Add new documents to the vector store
    vectorstore.add_documents(split_docs)
    vectorstore.save_local(VECTOR_DB_PATH)

    print(f"Added {len(docs_to_add)} new chunks to the vector store.")
    return vectorstore


if __name__ == "__main__":
    vectorstore = load_vector_store()

    documents = load_documents("resumes")
    if vectorstore is None:
        print("No vector store found. Building a new one from all resumes...")
        vectorstore = build_vector_store(documents)
    else:
        # Add only new resumes (avoid duplicates)
        vectorstore = add_new_documents(vectorstore, documents)

    print("\n=== Vector Store Sanity Check ===")
    total_docs = len(vectorstore.docstore._dict)
    print(f"Total documents in vector store: {total_docs}")

    # Sample preview
    print("\nSample documents:")
    for i, doc_id in enumerate(list(vectorstore.docstore._dict.keys())[:5], 1):
        doc = vectorstore.docstore._dict[doc_id]
        source = doc.metadata.get("source", "Unknown")
        snippet = doc.page_content[:150].replace("\n", " ")

        # Get the vector for this document
        vector = vectorstore.index.reconstruct(i - 1)  # FAISS stores vectors in the index
        vector_size = len(vector)

        print(f"{i}. Source: {source}")
        print(f"   Preview: {snippet}...")
        print(f"   Vector length: {vector_size}")
        print(f"   First 10 vector values: {vector[:10]}")
        print("-" * 50)
