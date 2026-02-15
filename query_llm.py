from collections import defaultdict
from dotenv import load_dotenv
from semantic_search import hybrid_rank
from populate_db import load_vector_store
# https://docs.langchain.com/oss/python/integrations/chat/huggingface
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.messages import HumanMessage, SystemMessage

# Load HF token from .env
load_dotenv()

# --------- RAG TUNING PARAMETERS ----------
TOP_K_CHUNKS = 15
TOP_N_CANDIDATES = 3
MAX_CHUNKS_PER_CANDIDATE = 2
# ------------------------------------------

def initialize_llm():
    hf_llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-alpha",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        provider="auto"  # Let HF select best provider
    )
    return ChatHuggingFace(llm=hf_llm)

def rank_candidates(job_desc):
    vectorstore = load_vector_store()
    llm = initialize_llm()

    # 1️⃣ Hybrid retrieval
    top_chunks = hybrid_rank(job_desc, vectorstore, alpha=0.5, top_k=TOP_K_CHUNKS)

    # 2️⃣ Group chunks by candidate
    candidate_chunks = defaultdict(list)
    for source, text, score in top_chunks:
        candidate_chunks[source].append((text, score))

    # 3️⃣ Compute top-N candidates by best hybrid score
    candidate_scores = []
    for source, chunks in candidate_chunks.items():
        best_score = max(score for _, score in chunks)
        candidate_scores.append((source, best_score))
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidate_scores[:TOP_N_CANDIDATES]

    print(f"Estimated Zephyr calls: {len(top_candidates)}")

    results = []

    # 4️⃣ Run LLM for top candidates
    for source, hybrid_score in top_candidates:
        # Select top chunks per candidate
        chunks = sorted(candidate_chunks[source], key=lambda x: x[1], reverse=True)[:MAX_CHUNKS_PER_CANDIDATE]
        combined_text = "\n\n".join(chunk for chunk, _ in chunks)

        messages = [
            SystemMessage(content="You are an expert recruiter."),
            HumanMessage(content=f"""
Job Description:
{job_desc}

Candidate Resume Excerpts:
{combined_text}

Tasks:
1. Provide a candidate ranking score (1-5)
2. List matched skills
3. Provide experience fit summary
4. Identify any gaps
Please format your answer clearly.
""")
        ]

        response = llm.invoke(messages)

        results.append({
            "candidate": source,
            "hybrid_score": round(float(hybrid_score), 3),
            "llm_analysis": response.content if hasattr(response, "content") else str(response)
        })

    return results

if __name__ == "__main__":
    job_desc = """
Looking for a Cost and Management Accountant (CMA) with auditing, MIS reporting,
budgeting experience, and familiarity with financial regulations.
"""
    results = rank_candidates(job_desc)
    for i, res in enumerate(results, 1):
        print(f"{i}. Candidate: {res['candidate']}")
        print(f"   Hybrid Score: {res['hybrid_score']}")
        print(f"   LLM Output:\n{res['llm_analysis']}")
        print("-" * 50)
