from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

def ingest_pdf(file_path, filename):

    index.delete(delete_all=True)

    loader = PyPDFLoader(file_path)

    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = filename

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts
    ).tolist()

    vectors = []

    for i, chunk in enumerate(chunks):

        vectors.append((
            str(i),
            embeddings[i],
            {
                "text": chunk.page_content,
                "source": filename
            }
        ))

    index.upsert(vectors=vectors)

    print("Embeddings stored successfully")