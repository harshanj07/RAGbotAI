import os
import requests
import faiss
import pickle
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import streamlit as st
import torch
from transformers import pipeline

VECTOR_DIM = 384
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
device = 0 if torch.cuda.is_available() else -1
SUMMARIZER = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e", device=device)
INDEX_FILE = "data/faiss.index"
DOC_EMBED_FILE = "data/docs.pkl"
URLS_FILE = "data/urls.pkl"
TITLE_FILE = "data/titles.pkl"
os.makedirs("data", exist_ok=True)

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def load_pickle(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except (EOFError, FileNotFoundError):
        return {} if "titles" in file_path else set() if "urls" in file_path else []

if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    index = faiss.IndexFlatL2(VECTOR_DIM)


documents = load_pickle(DOC_EMBED_FILE)
scraped_urls = load_pickle(URLS_FILE)
page_titles = load_pickle(TITLE_FILE)

def clean_text(text):
    lines = text.split("\n")
    filtered_lines = [line.strip() for line in lines if len(line.strip()) > 10]
    return "\n".join(filtered_lines)

def extract_text_chunks(text, chunk_size=1500, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def embed_texts(texts):
    return EMBEDDING_MODEL.encode(texts).tolist()

def persist_index():
    faiss.write_index(index, INDEX_FILE)
    with open(DOC_EMBED_FILE, "wb") as f:
        pickle.dump(documents, f)
    with open(URLS_FILE, "wb") as f:
        pickle.dump(scraped_urls, f)
    with open(TITLE_FILE, "wb") as f:
        safe_titles = {}
        for url, title in page_titles.items():
            try:
                safe_titles[url] = str(title)
            except Exception as e:
                print(f"[!] Failed to stringify title for {url}: {e}")
                safe_titles[url] = "Untitled Page"
        pickle.dump(safe_titles, f)




def fetch_url_text(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled Page"
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        return title, clean_text(" ".join(paragraphs))
    except Exception as e:
        return "Failed to fetch URL", f"Failed to fetch URL: {e}"



def summarize_text(text):
    if len(text) > 1000:
        summary = SUMMARIZER(text, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    return text

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="WebRAG", layout="centered")
st.title("\U0001F310 WebRAG")
st.caption("Scrape and query web content efficiently")

st.subheader("\U0001F517 Add URLs to Scrape")
url_input = st.text_input("Enter a URL")

if url_input:
    if url_input in scraped_urls:
        st.warning("⚠️ This URL has already been scraped.")
    else:
        with st.status("Fetching content from URL..."):
            title, url_text = fetch_url_text(url_input)
            if url_text.strip():
                url_chunks = extract_text_chunks(url_text)
                url_embeds = embed_texts(url_chunks)
                index.add(np.array(url_embeds).astype("float32"))
                documents.extend(url_chunks)
                scraped_urls.add(url_input)
                page_titles[url_input] = title
                persist_index()
                st.success(f"✅ URL content added: {title}")
            else:
                st.warning("❌ No relevant text found at the URL.")

if scraped_urls:
    st.subheader("\U0001F4DC Scraped URLs")
    for url in scraped_urls:
        st.write(f"🔹 {page_titles.get(url, 'Untitled Page')} - {url}")
        if st.button(f"🗑️ Delete {page_titles.get(url, 'Untitled Page')}", key=url):
            scraped_urls.remove(url)
            page_titles.pop(url, None)
            persist_index()
            st.rerun()

st.subheader("\U0001F4DD Filter/Search Indexed Content")
search_query = st.text_input("Search indexed content")

if search_query:
    query_vec = EMBEDDING_MODEL.encode([search_query])
    D, I = index.search(np.array(query_vec).astype("float32"), k=5)
    top_chunks = [documents[i] for i in I[0] if i < len(documents)]
    
    if top_chunks:
        st.markdown("### 🔍 Summarized Matches")
        for chunk in top_chunks:
            try:
                summary = summarize_text(chunk)
                st.write(summary)
            except Exception as e:
                st.write("❌ Error summarizing chunk:", e)
    else:
        st.warning("No relevant results found.")


st.subheader("\U0001F4AC Ask a question based on the scraped content")
if user_input := st.chat_input("Ask me anything!"):
    st.session_state.chat_history.append({"role": "user", "message": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Processing query..."):
            if len(documents) == 0:
                st.warning("⚠️ Please scrape at least one URL before querying.")
            else:
                query_vec = EMBEDDING_MODEL.encode([user_input])
                D, I = index.search(np.array(query_vec).astype("float32"), k=3)

                top_chunks = [documents[i] for i, score in zip(I[0], D[0]) if i < len(documents) and score < 1.0]

                if not top_chunks:
                    st.warning("❌ No relevant information found in the scraped content.")
                else:
                    context = "\n\n".join(top_chunks)

                    prompt = f"""
You are an intelligent assistant. Answer the user's question **only based on the provided context below**. 
If the answer is not present in the context, respond with "I couldn't find relevant information in the provided content."

Context:
\"\"\"
{context}
\"\"\"

Question: {user_input}
Answer:
                    """

                    response = requests.post(OLLAMA_API_URL, json={
                        "model": "mistral",
                        "prompt": prompt.strip(),
                        "stream": False
                    })

                    if response.status_code == 200:
                        answer = response.json().get("response", "No response").strip()
                    else:
                        answer = "Error fetching response from Ollama."

                    st.write(answer)
                    st.session_state.chat_history.append({"role": "assistant", "message": answer})
