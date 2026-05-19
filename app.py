from fastapi import FastAPI
from pydantic import BaseModel
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI()

# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# LLM Pipeline
llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=200
)

# Request schema
class QueryRequest(BaseModel):
    question: str


# Health route
@app.get("/")
def home():
    return {"message": "RAG Chatbot Running"}


# Question answering route
@app.post("/ask")
def ask_question(request: QueryRequest):

    # Convert question into embeddings
    question_embedding = embedding_model.encode(
        request.question
    ).tolist()

    # Search Pinecone
    results = index.query(
        vector=question_embedding,
        top_k=3,
        include_metadata=True
    )

    # Extract context
    context = "\n".join([
        match["metadata"]["text"]
        for match in results["matches"]
    ])

    # Prompt
    prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{request.question}

Answer:
"""
    response = llm(prompt)

    return {
        "answer": response[0]["generated_text"]
    }