from fastapi import FastAPI,UploadFile,File
from pydantic import BaseModel
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from dotenv import load_dotenv
import os
import shutil
from fastapi.middleware.cors import CORSMiddleware
from ingest import ingest_pdf

# Load environment variables
load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    max_new_tokens=200,
    temperature=0.3
)

# Request schema
class QueryRequest(BaseModel):
    question: str


# Health route
@app.get("/")
def home():
    return {"message": "RAG Chatbot Running"}


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
    sources = []
    for match in results["matches"]:
        metadata = match["metadata"]
        source = metadata.get("source", "Unknown")
        sources.append(source)
    # Prompt
    prompt = f"""
You are a helpful AI assistant.

Answer the question ONLY using the provided document context.

If the answer is not available in the document, say:
"I could not find relevant information in the uploaded PDF."
Context:
{context}

Question:
{request.question}

Answer:
"""
    response = llm(prompt,return_full_text=False)
    answer = response[0]["generated_text"].strip()

    return {
        "answer": answer
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    os.makedirs("data", exist_ok=True)

    # DELETE OLD PDFS
    for old_file in os.listdir("data"):

        if old_file.endswith(".pdf"):
            os.remove(f"data/{old_file}")

    # SAVE NEW PDF
    file_path = f"data/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # INGEST PDF
    ingest_pdf(file_path, file.filename)

    return {
        "message": f"{file.filename} uploaded and indexed successfully"
    }