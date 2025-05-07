import streamlit as st
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time
import os

# --- Page Configuration (MUST BE FIRST) ---
st.set_page_config(layout="wide", page_title="Arabic Search Interface")

# --- Configuration ---
PARAGRAPHS_FILE = "Paragraphs.txt"
FAISS_INDEX_FILE = "faiss_index.bin"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_N = 5

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
        model = SentenceTransformer(model_name, device='cpu')  # Force CPU to avoid CUDA conflicts
        return model
    except Exception as e:
        st.error(f"🚨 Error loading Sentence Transformer model '{model_name}': {e}")
        return None

@st.cache_resource
def setup_tfidf_vectorizer(paragraphs):
    """Initializes and fits a TF-IDF vectorizer with preprocessing."""
    if not paragraphs:
        st.warning("⚠ Cannot set up TF-IDF: No paragraphs loaded.")
        return None, None
    try:
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(paragraphs)
        return vectorizer, tfidf_matrix
    except Exception as e:
        st.error(f"🚨 An error occurred setting up TF-IDF: {e}")
        return None, None

# --- Search Functions ---

def perform_classical_search(query, vectorizer, tfidf_matrix, paragraphs, top_n):
    """Performs TF-IDF search."""
    if not query or vectorizer is None or tfidf_matrix is None or not paragraphs:
        st.warning("⚠ Invalid input for classical search: Check query or resources.")
        return [], [], 0
    try:
        start_time = time.time()
        query_vector = vectorizer.transform([query.strip()])
        if query_vector.nnz == 0:  # Check if query vector is empty
            st.warning("⚠ Query contains no valid terms for TF-IDF search.")
            return [], [], 0
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_indices = np.argsort(cosine_similarities)[::-1][:top_n]
        search_time = time.time() - start_time
        results = [paragraphs[i] for i in top_indices if i < len(paragraphs)]
        scores = [cosine_similarities[i] for i in top_indices if i < len(paragraphs)]
        return results, scores, search_time
    except Exception as e:
        st.error(f"🚨 Classical search error: {e}")
        return [], [], 0

def perform_semantic_search(query, model, index, paragraphs, top_n):
    """Performs embedding-based semantic search using FAISS."""
    if not query or model is None or index is None or not paragraphs:
        st.warning("⚠ Invalid input for semantic search: Check query or resources.")
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

# --- Streamlit App UI ---

st.sidebar.image("book_cover.jpg", caption="📖 Arabic Search Interface", use_container_width=True)
st.title("🔎 Arabic Retrieval System")
st.markdown("Enter your Arabic query below to get results from Classical (TF-IDF) and Semantic (Embeddings) search.")
st.divider()

# --- Load data and initialize models ---
paragraphs = load_paragraphs(PARAGRAPHS_FILE)
faiss_index = load_faiss_index(FAISS_INDEX_FILE)
tfidf_vectorizer, corpus_tfidf_matrix = setup_tfidf_vectorizer(paragraphs)
embedding_model = load_sentence_transformer_model(EMBEDDING_MODEL)

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

if tfidf_vectorizer is None or corpus_tfidf_matrix is None:
    st.error("🚨 Failed to initialize TF-IDF vectorizer.")
    load_error = True
#else:
    #st.sidebar.success("✅ TF-IDF vectorizer ready.")

if embedding_model is None:
    st.error("🚨 Failed to load embedding model.")
    load_error = True
#else:
    #st.sidebar.success("✅ Embedding model loaded.")

if load_error:
    st.warning("⚠ Cannot proceed due to resource loading errors. Check messages above.")
    st.stop()

# --- User Input ---
query = st.text_input("📝 أدخل استعلام البحث هنا (Enter Search Query Here):", key="query_input", placeholder="مثال: من هم العرب؟")

# --- Perform Search and Display Results ---
if query:
    st.divider()
    st.subheader(f"🔍 نتائج البحث عن: \"{query}\"")
    col1, col2 = st.columns(2)

    # --- Classical Search ---
    with col1:
        st.markdown("### (Classical Search - TF-IDF)")
        with st.spinner("⏳ Loading classical search ..."):
            classical_results, classical_scores, classical_time = perform_classical_search(
                query, tfidf_vectorizer, corpus_tfidf_matrix, paragraphs, TOP_N
            )
        st.caption(f"⏱ Time: {classical_time:.4f} seconds")
        if classical_results:
            for i, (res, score) in enumerate(zip(classical_results, classical_scores)):
                st.markdown(f"""
                <div style="text-align: right; border: 1px solid #eee; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <b>{i+1}. (Score: {score:.4f})</b><br>
                    {res}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("لم يتم العثور على نتائج للبحث الكلاسيكي.")

    # --- Semantic Search ---
    with col2:
        st.markdown("### (Semantic Search - Embeddings)")
        with st.spinner("⏳ Loading semantic search ..."):
            semantic_results, semantic_scores, semantic_time = perform_semantic_search(
                query, embedding_model, faiss_index, paragraphs, TOP_N
            )
        st.caption(f"⏱ Time: {semantic_time:.4f} seconds")
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

else:
    if not load_error:
        st.info("يرجى إدخال استعلام في المربع أعلاه لبدء البحث.")

st.divider()
st.caption(f"Powered by Streamlit | Classical Search: TF-IDF | Semantic Search: FAISS & '{EMBEDDING_MODEL}'")