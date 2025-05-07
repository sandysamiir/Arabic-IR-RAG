import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import time
import os

# --- Page Configuration (MUST BE FIRST) ---
st.set_page_config(layout="wide", page_title="Arabic RAG Interface")

# --- Configuration ---
PARAGRAPHS_FILE = "Paragraphs.txt"
FAISS_INDEX_FILE = "faiss_index.bin"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "gemini-2.0-flash"
TOP_N = 5

# --- Load API Key ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("🚨 Google API key not found. Set the GOOGLE_API_KEY environment variable.")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)

# --- Caching Functions ---

@st.cache_data
def load_paragraphs(filepath):
    """Loads paragraphs from a text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            paragraphs = [line.strip() for line in f if line.strip()]
        if not paragraphs:
            st.error(f"🚨 No paragraphs found in {filepath}.")
            return None
        return paragraphs
    except FileNotFoundError:
        st.error(f"🚨 Error: Paragraphs file not found at {filepath}")
        return None
    except Exception as e:
        st.error(f"🚨 An error occurred loading paragraphs: {e}")
        return None

@st.cache_resource
def load_faiss_index(filepath):
    """Loads a pre-built FAISS index."""
    try:
        index = faiss.read_index(filepath)
        return index
    except FileNotFoundError:
        st.error(f"🚨 Error: FAISS index file not found at {filepath}")
        return None
    except Exception as e:
        st.error(f"🚨 An error occurred loading the FAISS index: {e}")
        return None

@st.cache_resource
def load_sentence_transformer_model(model_name):
    """Loads a Sentence Transformer model."""
    try:
        model = SentenceTransformer(model_name, device='cpu')
        return model
    except Exception as e:
        st.error(f"🚨 Error loading Sentence Transformer model '{model_name}': {e}")
        return None

@st.cache_resource
def load_gemini_model(model_name):
    """Loads the Gemini LLM model."""
    try:
        model = genai.GenerativeModel(model_name)
        return model
    except Exception as e:
        st.error(f"🚨 Error loading Gemini model '{model_name}': {e}")
        return None

# --- Search and LLM Functions ---

def perform_semantic_search(query, model, index, paragraphs, top_n):
    """Performs embedding-based semantic search using FAISS."""
    if not query or model is None or index is None or not paragraphs:
        st.warning("⚠️ Invalid input for semantic search: Check query or resources.")
        return [], [], 0
    try:
        start_time = time.time()
        query_embedding = model.encode([query.strip()])
        query_embedding_np = np.array(query_embedding).astype('float32')
        if query_embedding_np.ndim == 1:
            query_embedding_np = np.expand_dims(query_embedding_np, axis=0)
        distances, indices = index.search(query_embedding_np, top_n)
        search_time = time.time() - start_time
        results = [paragraphs[i] for i in indices[0] if i != -1 and i < len(paragraphs)]
        scores = [1 / (1 + d) for d in distances[0] if d >= 0]
        return results, scores, search_time
    except Exception as e:
        st.error(f"🚨 Semantic search error: {e}")
        return [], [], 0

def generate_rag_response(query, context_paragraphs, llm_model):
    """Generates a response using Gemini with query and context."""
    if not query or not context_paragraphs or llm_model is None:
        st.warning("⚠️ Invalid input for RAG response.")
        return "No response generated."
    try:
        context = "\n\n".join([f"Paragraph {i+1}: {para}" for i, para in enumerate(context_paragraphs)])
        prompt = f"""You are an expert assistant answering in Arabic. Use the following context to provide a concise and accurate answer to the query. Do not mention the context explicitly in your response.

**Query**: {query}

**Context**:
{context}

**Answer**: """
        response = llm_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"🚨 RAG response error: {e}")
        return "Error generating RAG response."

def generate_query_only_response(query, llm_model):
    """Generates a response using Gemini with query only."""
    if not query or llm_model is None:
        st.warning("⚠️ Invalid input for query-only response.")
        return "No response generated."
    try:
        prompt = f"""You are an expert assistant answering in Arabic. Provide a concise and accurate answer to the following query without any additional context.

**Query**: {query}

**Answer**: """
        response = llm_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"🚨 Query-only response error: {e}")
        return "Error generating query-only response."

# --- Streamlit App UI ---
st.sidebar.image("book_cover.jpg", caption="📖 Arabic Search Interface", use_container_width=True)
st.title("🔎 Arabic RAG System")
st.markdown("Enter your Arabic query below to get semantic search results and intelligent answers using Gemini 2.0 Flash.")
st.divider()

# --- Load data and initialize models ---
paragraphs = load_paragraphs(PARAGRAPHS_FILE)
faiss_index = load_faiss_index(FAISS_INDEX_FILE)
embedding_model = load_sentence_transformer_model(EMBEDDING_MODEL)
llm_model = load_gemini_model(LLM_MODEL)

# Check for loading errors
load_error = False
if paragraphs is None:
    st.error("🚨 Failed to load paragraphs.")
    load_error = True
#else:
    #st.sidebar.success(f"✅ Paragraphs loaded ({len(paragraphs)}).")

if faiss_index is None:
    st.error("🚨 Failed to load FAISS index.")
    load_error = True
#else:
    #st.sidebar.success("✅ FAISS index loaded.")

if embedding_model is None:
    st.error("🚨 Failed to load embedding model.")
    load_error = True
#else:
    #st.sidebar.success("✅ Embedding model loaded.")

if llm_model is None:
    st.error("🚨 Failed to load Gemini model.")
    load_error = True
#else:
    #st.sidebar.success("✅ Gemini model loaded.")

if load_error:
    st.warning("⚠️ Cannot proceed due to resource loading errors. Check messages above.")
    st.stop()

# --- User Input ---
query = st.text_input("📝 أدخل استعلام البحث هنا (Enter Search Query Here):", key="query_input", placeholder="مثال: ما هو دور العلماء المسلمين في النهضة الأوروبية؟")

# --- Perform Search and Display Results ---
if query:
    st.divider()
    st.subheader(f"🔍 نتائج البحث والإجابات لـ: \"{query}\"")
    
    # --- Semantic Search ---
    st.markdown("### (Semantic Search Results)")
    with st.spinner("⏳ Loady semantic search ..."):
        semantic_results, semantic_scores, semantic_time = perform_semantic_search(
            query, embedding_model, faiss_index, paragraphs, TOP_N
        )
    st.caption(f"⏱️ Search Time: {semantic_time:.4f} seconds")
    if semantic_results:
        for i, (res, score) in enumerate(zip(semantic_results, semantic_scores)):
            st.markdown(f"""
            <div style="text-align: right; border: 1px solid #eee; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <b>{i+1}. (Score: {score:.4f})</b><br>
                {res}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("لم يتم العثور على نتائج للبحث الدلالي.")

    # --- RAG Response ---
    st.markdown("### (RAG Response)")
    with st.spinner("⏳ Loading RAG response ..."):
        rag_response = generate_rag_response(query, semantic_results, llm_model)
    st.markdown(f"""
    <div style="text-align: right; border: 1px solid #ddd; padding: 15px; border-radius: 5px;">
        {rag_response}
    </div>
    """, unsafe_allow_html=True)

    # --- Query-Only Response ---
    st.markdown("### (Query-Only Response)")
    with st.spinner("⏳ Loading LLM response ..."):
        query_only_response = generate_query_only_response(query, llm_model)
    st.markdown(f"""
    <div style="text-align: right; border: 1px solid #ddd; padding: 15px; border-radius: 5px;">
        {query_only_response}
    </div>
    """, unsafe_allow_html=True)

else:
    if not load_error:
        st.info("يرجى إدخال استعلام في المربع أعلاه لبدء البحث.")

st.divider()
st.caption(f"Powered by Streamlit | Semantic Search: FAISS & '{EMBEDDING_MODEL}' | LLM: Gemini 2.0 Flash")