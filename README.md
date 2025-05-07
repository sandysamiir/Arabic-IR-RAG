# Arabic Information Retrieval and RAG System

This project demonstrates classical and semantic information retrieval (IR), as well as Retrieval-Augmented Generation (RAG), on an Arabic book. It supports Arabic queries and generates answers using LLMs. The entire system is built using FAISS, sentence embeddings, and Google Gemini (2.0 Flash).

## 📖 Book Used

- **Title:** أثر العرب في الحضارة الأوروبية  
- **Author:** عباس العقاد
- **Source:** [noor-book.com](https://www.noor-book.com)  
- **Format:** Arabic PDF, converted to text using OCR 

---

## 🗂 Project Structure

| File | Description |
|------|-------------|
| `أثر العرب في الحضارة الأوروبية.pdf` | The source Arabic book |
| `أثر العرب في الحضارة الأوروبية.txt` | OCR-converted plain text |
| `Paragraphs.txt` | Split version of the book (2–4 sentence paragraphs) |
| `paragraph_embeddings.npy` | Sentence embeddings |
| `faiss_index.bin` | FAISS index file |
| `book_cover.jpg` | Book cover image used in the interface |
| `Book_Preparation.ipynb` | Main Colab notebook to build and test the system |
| `Retrieval_app.py` | Streamlit interface for classical & semantic search |
| `RAG_app.py` | Streamlit interface for Q&A using RAG |

---

## 💻 Setup Instructions

### ▶️ Option 1: Run in Google Colab (Recommended)

1. Open `Book_Preparation.ipynb` in Google Colab.
2. Run each cell **in order**:
   - Install required libraries
   - Convert PDF to text
   - Split text into paragraphs
   - Generate embeddings
   - Index with FAISS
   - Run retrieval system
   - Set API Key
   - Run RAG system

3. **Where to change paths:**
   - In the **Convert PDF to Text** cell, update the path of the PDF file to its actual path or if another book is used:
     ```python
     file_path = '/content/أثر العرب في الحضارة الأوروبية.pdf'
     ```

   - Both `Retrieval_app.py` and `RAG_app.py` rely on the following lines being correct:
     ```python
     PARAGRAPHS_FILE = "Paragraphs.txt"
     FAISS_INDEX_FILE = "faiss_index.bin"
     ```
     Make sure these files are in the same directory or update the paths if you move or rename them.

4. **Set API Key (Gemini):**
   - In the `Set API Key` cell, paste your API key:
     ```python
     import os
     os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"
     ```
You may run any cell to test its functionality individually, as long as the required input files it depends on are already available in the current runtime.

---

### 💻 Option 2: Run Locally

1. Clone this repository and navigate to the project directory.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Change paths:**
   - In the **Convert PDF to Text** cell, update the path of the PDF file to its actual path or if another book is used:
     ```python
     file_path = '/content/أثر العرب في الحضارة الأوروبية.pdf'
     ```

   - Both `Retrieval_app.py` and `RAG_app.py` rely on the following lines being correct:
     ```python
     PARAGRAPHS_FILE = "Paragraphs.txt"
     FAISS_INDEX_FILE = "faiss_index.bin"
     ```
     Make sure these files are in the same directory or update the paths if you move or rename them.
   
4. **Place your Gemini API key:**
   - In the `Set API Key` cell, paste your API key:
     ```python
     import os
     os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"
     ```

5. To run the retrieval interface, run the following command
   ```python
     streamlit run Retrieval_app.py
     ```
6. To run the RAG (Q&A) interface, run the following command
   ```python
     streamlit run RAG_app.py
     ```
---

## 🔍 Features
- Classical Search: TF-IDF-based keyword search
- Semantic Search: SentenceTransformer embeddings using (paraphrase-multilingual-MiniLM-L12-v2) model + FAISS
- RAG Q&A: Google Gemini Flash 2.0 generates answers using retrieved context
- LLM-Only Q&A: For comparison without retrieval
- Arabic Language Support: All inputs and outputs are in Arabic
---

## 📦 Dependencies
See `requirements.txt` for the full list of libraries used, including:
- sentence-transformers
- faiss-cpu
- streamlit
- unstructured[pdf]
- google-generativeai
---

## 📌 Notes
- The book is in Arabic and preprocessed into paragraphs for better embedding and retrieval.
- You can try any Arabic question related to the book in both interfaces.
- You’ll see classical vs. semantic results side by side, and LLM-only vs. RAG answers for Q&A.
