# RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) chatbot built using FastAPI, Streamlit, Pinecone, Sentence Transformers, and TinyLlama.

The application allows users to upload PDFs, generate embeddings, store them in Pinecone, and ask context-aware questions from the uploaded document.

---

## Features

- PDF Upload
- Automatic Text Chunking
- Semantic Embeddings
- Pinecone Vector Database
- Context-Aware Question Answering
- Streamlit Frontend
- FastAPI Backend
- Automatic Vector Cleanup
- End-to-End RAG Pipeline

---

## Tech Stack

- FastAPI
- Streamlit
- Pinecone
- Sentence Transformers
- TinyLlama
- LangChain
- Docker

---

## Project Structure

```text
RAG_Chatbot/
│
├── app.py
├── ingest.py
├── frontend.py
├── requirements.txt
├── Dockerfile
├── data/
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/akashraj1822/RAG_Chatbot.git
cd RAG_Chatbot
```

### Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment Variables

Create a `.env` file:

```env
PINECONE_API_KEY=your_api_key
PINECONE_INDEX_NAME=rag-chatbot
```

---

## Run FastAPI Backend

```bash
uvicorn app:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

---

## Run Streamlit Frontend

```bash
streamlit run frontend.py
```

Frontend URL:

```text
http://localhost:8501
```

---

## Workflow

```text
Upload PDF
    ↓
Chunking & Embeddings
    ↓
Store in Pinecone
    ↓
Ask Questions
    ↓
Semantic Retrieval
    ↓
LLM Generated Answer
```

---

## Future Improvements

- Multi-PDF Support
- Streaming Responses
- Hybrid Search
- Chat Memory
- Authentication
- Better LLM Models
- Docker Deployment

---

## Author

Akash Raj

GitHub: https://github.com/akashraj1822/RAG_Chatbot
