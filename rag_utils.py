import os
import fitz
import boto3
from dotenv import load_dotenv

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Upload file to AWS S3
def upload_to_s3(file_obj, filename):
    try:
        s3 = boto3.client("s3")
        s3.upload_fileobj(file_obj, "mines3bucket25", filename)
    except Exception as e:
        raise RuntimeError(f"AWS S3 upload failed: {e}")

# Extract and chunk document
def process_file(file_path):
    try:
        loader = PyMuPDFLoader(file_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(pages)
    except Exception as e:
        raise RuntimeError(f"Text extraction or splitting failed: {e}")

# Create and persist ChromaDB vector store
def build_vector_store(docs):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        db = Chroma.from_documents(docs, embeddings, persist_directory="./database/chroma_db")
        db.persist()
        return db
    except Exception as e:
        raise RuntimeError(f"Embedding or ChromaDB storage failed: {e}")

# Retrieve and generate answer with Groq LLM
def run_query_with_rag(query):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        db = Chroma(persist_directory="./database/chroma_db", embedding_function=embeddings)
        retriever = db.as_retriever()
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=GROQ_API_KEY,
            temperature=0
        )
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        return chain.run(query)
    except Exception as e:
        raise RuntimeError(f"Groq RAG query failed: {e}")
