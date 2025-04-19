# 🧠 RAGbot — Generative AI with Web Scraping and Retrieval-Augmented Generation

RAGbot is a Python-based Generative AI project that leverages **Ollama**, the **Mistral model**, and web scraping to build a smart question-answering system based on web content. It scrapes content from a URL, stores it in chunks, and answers questions using the most relevant sections.

---

## 🚀 Features

- 🔗 Accepts a URL input (e.g., Wikipedia articles)
- 🕸️ Web scrapes the content using `BeautifulSoup`
- 📦 Chunks and stores content in `.pkl` format (pickle database)
- 🔍 Retrieves relevant content based on user keywords
- 🤖 Answers questions using the Mistral model via Ollama

---
<!-- 
## 📁 Folder Structure

```bash
RAGbot/
├── app.py                  # Main script to run the RAGbot
├── scraper.py              # Scrapes content from a given URL
├── chunker.py              # Splits content into chunks
├── retriever.py            # Handles keyword matching and chunk retrieval
├── qa_engine.py            # Uses Ollama + Mistral to answer questions
├── data/
│   └── chunks.pkl          # Stored chunks database
├── .env                    # Environment variables (if any)
├── requirements.txt        # Required Python packages
└── README.md               # You're here! -->
