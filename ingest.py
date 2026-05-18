from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Load PDF
loader = PyPDFLoader("data/sample.pdf")
docs = loader.load()

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(INDEX_NAME)

# Store embeddings
for i, chunk in enumerate(chunks):
    embedding = embedding_model.encode(chunk.page_content).tolist()

    index.upsert([
        (
            str(i),
            embedding,
            {"text": chunk.page_content}
        )
    ])

print("Embeddings stored successfully")