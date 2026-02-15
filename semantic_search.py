import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from populate_db import load_vector_store


def hybrid_rank(job_desc, vectorstore, alpha=0.5, top_k=10):
    """
    Rank resume chunks based on a hybrid of precomputed semantic embeddings + keyword similarity.

    Returns a list of tuples: (doc.metadata["source"], chunk_text, final_score)
    """
    # 1. Extract all chunk texts and embeddings from FAISS
    chunks = [doc.page_content for doc in vectorstore.docstore._dict.values()]  # original text to display, reason, or feed into LLM
    sources = [doc.metadata.get("source", "Unknown") for doc in vectorstore.docstore._dict.values()]  # resume file, so we can trace back the chunk to the original resume
    embeddings = [vectorstore.index.reconstruct(i) for i in range(len(chunks))]  # Precomputed embeddings are stored in FAISS index

    # 2️. Embed job description
    embeddings_model = vectorstore.embedding_function
    job_embedding = embeddings_model.embed_query(job_desc)

    # 3️.1 Semantic similarity
    # these are pre-computed in FAISS but since we combine with TFIDF we need to reconstruct these
    semantic_scores = [cosine_similarity([job_embedding], [chunk_vec])[0][0] for chunk_vec in embeddings]

    # 3.2 Keyword similarity (TF-IDF)
    vectorizer = TfidfVectorizer()
    texts = [job_desc] + chunks
    tfidf_matrix = vectorizer.fit_transform(texts)
    keyword_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # 4. Weighted hybrid score
    semantic_scores = np.array(semantic_scores)
    keyword_scores = np.array(keyword_scores)
    # Optional: normalize both to [0,1] for fair weighting
    semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)
    keyword_scores = (keyword_scores - keyword_scores.min()) / (keyword_scores.max() - keyword_scores.min() + 1e-8)

    final_scores = alpha * semantic_scores + (1 - alpha) * keyword_scores

    # Finally, Return top_k results
    top_indices = np.argsort(final_scores)[-top_k:][::-1]  # descending
    results = [(sources[i], chunks[i], final_scores[i]) for i in top_indices]

    return results


if __name__ == "__main__":
    vectorstore = load_vector_store()
    # job_desc = "Looking for a Cost and Management Accountant (CMA) with auditing, MIS reporting, and budgeting experience."
    job_desc = "Looking for an Electrical Engineer with circuit design, system testing, and compliance auditing experience."
    top_chunks = hybrid_rank(job_desc, vectorstore, alpha=0.5, top_k=5)

    for i, (source, text, score) in enumerate(top_chunks, 1):
        # Ensure text is string
        if isinstance(text, list):
            text = " ".join(text)
        elif not isinstance(text, str):
            text = str(text)

        snippet = text[:200].replace("\n", " ").replace("\r", " ")

        print(f"{i}. Source: {source}")
        print(f"   Score: {score:.4f}")
        print(f"   Snippet: {snippet}...")
        print("-" * 50)
