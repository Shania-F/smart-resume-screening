import streamlit as st
from query_llm import rank_candidates

st.set_page_config(page_title="SMART RESUME SCREENING", layout="wide")

st.title("SMART RESUME SCREENING")
st.markdown("Upload a Job Description and instantly rank candidates.")

job_desc = st.text_area("Paste Job Description Here", height=200)

if st.button("Rank Candidates"):
    if not job_desc.strip():
        st.warning("Please enter a job description.")
    else:
        with st.spinner("Analyzing resumes..."):
            try:
                results = rank_candidates(job_desc)
            except Exception as e:
                st.error(f"Error: {e}")
                results = []

        if results:
            st.success("Ranking Complete!")
            for i, res in enumerate(results, 1):
                with st.expander(f"{i}. {res['candidate']} (Hybrid Score: {res['hybrid_score']})"):
                    st.write(res["llm_analysis"])
        else:
            st.info("No results returned.")
