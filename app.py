import streamlit as st
import uuid
from io import BytesIO
import os
from dotenv import load_dotenv

from rag_utils import (
    upload_to_s3,
    process_file,
    build_vector_store,
    run_query_with_rag
)

# Setup
st.set_page_config(page_title="LangChain RAG Assistant", layout="wide")
st.title("📄 Advanced Documented RAG Assistant")
load_dotenv()

# File Upload
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    try:
        # Cache the file in memory
        file_bytes = uploaded_file.read()
        filename = f"{uuid.uuid4()}_{uploaded_file.name}"

        # Upload to S3
        with st.spinner("Uploading to S3..."):
            upload_to_s3(BytesIO(file_bytes), filename)
            st.success("✅ File uploaded successfully to S3")

        # Save temporarily to disk
        local_path = f"temp_{uploaded_file.name}"
        with open(local_path, "wb") as f:
            f.write(file_bytes)

        # Process, split, embed, and store
        with st.spinner("Processing and indexing the document..."):
            docs = process_file(local_path)
            build_vector_store(docs)
            st.success("✅ Document embedded and stored in ChromaDB")

    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()

# Query Section
st.subheader("🔍 Ask a question about your uploaded document")

query = st.text_input("Enter your question")

if query:
    try:
        with st.spinner("Generating answer with Groq LLM..."):
            answer = run_query_with_rag(query)
            st.markdown("### 💡 Answer")
            st.write(answer)
    except Exception as e:
        st.error(f"❌ {e}")
