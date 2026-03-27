"""
src/dashboard/app.py — MediRAG Streamlit Dashboard
===================================================
Visual interface for the evaluation pipeline.
Allows users to input a medical question, configure LLM settings 
(Provider, API Key, Model), and view the end-to-end evaluation results.

Usage:
    streamlit run src/dashboard/app.py
"""

import json
import time
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page Config & Styles
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MediRAG | Evaluation Dashboard",
    page_icon="🏥",
    layout="wide",
)

st.markdown("""
<style>
.metric-box {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
    border-left: 5px solid #007bff;
}
.metric-title { font-size: 14px; color: #6c757d; font-weight: bold; text-transform: uppercase; }
.metric-value { font-size: 24px; font-weight: bold; color: #343a40; }
.annot-entailed { background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 3px; }
.annot-contradicted { background-color: #f8d7da; color: #721c24; padding: 2px 4px; border-radius: 3px; font-weight: bold; }
.annot-neutral { background-color: #e2e3e5; color: #383d41; padding: 2px 4px; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Init
# ---------------------------------------------------------------------------
if "eval_result" not in st.session_state:
    st.session_state.eval_result = None
if "eval_latency" not in st.session_state:
    st.session_state.eval_latency = 0


# ---------------------------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Evaluation Engine")
    api_url = st.text_input("API URL", value="http://localhost:8000")

    st.subheader("LLM Settings")
    provider_choice = st.selectbox("Provider", ["Gemini", "Ollama"])

    if provider_choice == "Gemini":
        llm_api_key = st.text_input("Gemini API Key", type="password", help="Sent per-request, never saved.")
        llm_model = st.selectbox("Model", ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"])
        ollama_url = None
    else:
        llm_api_key = None
        llm_model = st.text_input("Ollama Model", value="mistral")
        ollama_url = st.text_input("Ollama Base URL", value="http://localhost:8080")

    st.subheader("Retrieval Settings")
    top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=5)
    run_ragas = st.checkbox("Run RAGAS (slower)", value=False)


# ---------------------------------------------------------------------------
# Main Layout
# ---------------------------------------------------------------------------
st.title("🏥 MediRAG Eval Dashboard")
st.markdown("End-to-End Pipeline: Retrieval → LLM Generation → Verification")

# --- Input Area ---
question_input = st.text_area(
    "Medical Question",
    height=100,
    placeholder="e.g., What is the recommended dosage of Metformin for elderly Type 2 Diabetes patients?",
)

if st.button("🚀 Run Pipeline", type="primary"):
    if not question_input.strip():
        st.warning("Please enter a medical question.")
    elif provider_choice == "Gemini" and not llm_api_key:
        st.warning("Please enter your Gemini API Key in the sidebar.")
    else:
        payload = {
            "question": question_input.strip(),
            "top_k": top_k,
            "run_ragas": run_ragas,
            "llm_provider": provider_choice.lower(),
        }
        if llm_api_key:
            payload["llm_api_key"] = llm_api_key
        if llm_model:
            payload["llm_model"] = llm_model
        if ollama_url:
            payload["ollama_url"] = ollama_url

        with st.spinner(f"Retrieving → Generating ({provider_choice}) → Evaluating..."):
            t0 = time.time()
            try:
                resp = requests.post(f"{api_url}/query", json=payload, timeout=125)
                resp.raise_for_status()
                st.session_state.eval_result = resp.json()
            except requests.exceptions.RequestException as e:
                err_text = str(e)
                if getattr(e, "response", None) is not None:
                    try:
                        err_text = e.response.json().get("detail", err_text)
                    except ValueError:
                        err_text = e.response.text
                st.error(f"API Error: {err_text}")
                st.session_state.eval_result = None
            
            st.session_state.eval_latency = round(time.time() - t0, 1)

# ---------------------------------------------------------------------------
# Results Display
# ---------------------------------------------------------------------------
res = st.session_state.eval_result
if res:
    st.divider()

    # --- Header Metrics ---
    cols = st.columns(4)
    hrs = res.get("hrs", 0)
    rb = res.get("risk_band", "UNKNOWN")
    
    # Color logic for HRS
    hrs_color = "#28a745" if hrs < 40 else "#ffc107" if hrs < 70 else "#fd7e14" if hrs < 85 else "#dc3545"

    with cols[0]:
        st.markdown(f"""
        <div class="metric-box" style="border-left-color: {hrs_color}">
            <div class="metric-title">Health Risk Score</div>
            <div class="metric-value" style="color: {hrs_color}">{hrs} / 100</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Risk Band</div>
            <div class="metric-value">{rb}</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Composite</div>
            <div class="metric-value">{res.get('composite_score', 0):.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Latency</div>
            <div class="metric-value">{st.session_state.eval_latency}s</div>
        </div>
        """, unsafe_allow_html=True)

    # --- Generated Answer ---
    st.subheader("🤖 Generated Answer")
    st.info(res.get("generated_answer", ""))

    # --- Module Scores Split ---
    st.markdown("### 📊 Module Breakdown")
    colA, colB = st.columns(2)
    mods = res.get("module_results", {})

    def show_mod(title: str, mod_data: Dict[str, Any], col):
        if not mod_data:
            return
        score = mod_data.get("score", 0.0)
        err = mod_data.get("error")
        lbl = f"{title} — {score:.2f}" if not err else f"⚠️ {title} (Error)"
        col.progress(score, text=lbl)
        if err:
            col.caption(f"Error: {err}")

    with colA:
        show_mod("Faithfulness", mods.get("faithfulness", {}), colA)
        show_mod("Source Credibility", mods.get("source_credibility", {}), colA)
    with colB:
        show_mod("Contradiction", mods.get("contradiction", {}), colB)
        if "ragas" in mods and mods["ragas"]:
            show_mod("RAGAS Answer Rel.", mods.get("ragas", {}), colB)

    # --- Deep Dive Tabs ---
    st.markdown("### 🔍 Deep Dive")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Claims & Faithfulness",
        "Entities",
        "Contradictions",
        "Retrieved Sources",
        "Raw JSON"
    ])

    # Tab 1: Faithfulness
    with tab1:
        f_details = mods.get("faithfulness", {}).get("details", {})
        claims = f_details.get("claims", [])
        if not claims:
            st.write("No claims extracted from answer.")
        else:
            for i, cl in enumerate(claims, 1):
                status = cl.get("status", "UNKNOWN")
                css_class = {
                    "ENTAILED": "annot-entailed",
                    "CONTRADICTED": "annot-contradicted",
                    "NEUTRAL": "annot-neutral"
                }.get(status, "annot-neutral")
                
                st.markdown(
                    f"**{i}.** <span class='{css_class}'>{status}</span> (NLI: {cl.get('nli_score', 0):.3f})<br>"
                    f"<i>\"{cl.get('claim')}\"</i>",
                    unsafe_allow_html=True
                )
                st.caption(f"Best chunk: `{cl.get('best_chunk_id')}`")
                st.markdown("---")

    # Tab 2: Entities
    with tab2:
        e_details = mods.get("entity_verifier", {}).get("details", {})
        entities = e_details.get("entities", [])
        if not entities:
            st.write("No medical entities found in answer.")
        else:
            df = pd.DataFrame(entities)
            # Reorder columns for readability if they exist
            cols_order = ["entity", "type", "status", "context_value", "rxcui"]
            avail_cols = [c for c in cols_order if c in df.columns]
            other_cols = [c for c in df.columns if c not in avail_cols]
            st.dataframe(df[avail_cols + other_cols], use_container_width=True)

    # Tab 3: Contradictions
    with tab3:
        c_details = mods.get("contradiction", {}).get("details", {})
        pairs = c_details.get("pairs", [])
        flagged = [p for p in pairs if p.get("flagged")]
        
        if not pairs:
            st.write("No contradiction pairs analyzed.")
        elif not flagged:
            st.success(f"Checked {len(pairs)} sentence pairs. Zero contradictions found! 🎉")
        else:
            st.error(f"Found {len(flagged)} potential contradictions.")
            for p in flagged:
                st.markdown(f"**Answer Sentence:** {p.get('sentence_a')}")
                st.markdown(f"**Context Sentence:** {p.get('sentence_b')}")
                st.caption(f"Contradiction Score: {p.get('contradiction_score', 0):.3f}")
                st.markdown("---")

    # Tab 4: Sources
    with tab4:
        chunks = res.get("retrieved_chunks", [])
        s_details = mods.get("source_credibility", {}).get("details", {}).get("chunks", [])
        
        # Merge chunk texts with tier info
        tier_db = {c["chunk_id"]: c for c in s_details}
        
        for i, ck in enumerate(chunks, 1):
            tier_info = tier_db.get(ck["chunk_id"], {})
            tier_num = tier_info.get("tier", "?")
            tier_type = tier_info.get("tier_type", "unknown")
            score = ck.get("similarity_score", 0.0)
            
            with st.expander(f"Source {i} | Tier {tier_num} ({tier_type}) | FAISS={score:.3f}"):
                st.markdown(f"**Title:** {ck.get('title')}")
                st.markdown(f"**Type:** {ck.get('pub_type')} | **Year:** {ck.get('pub_year')}")
                st.text_area("Text", ck.get('text', ''), height=100, disabled=True, key=f"txt_{i}")

    # Tab 5: Raw JSON
    with tab5:
        st.json(res)
