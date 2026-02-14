from fastapi import FastAPI, UploadFile, File
from sentence_transformers import SentenceTransformer
import os
import numpy as np

from utils.pdf_loader import extract_text_from_pdf
from utils.text_chunker import chunk_text
from embeddings import get_embedding
from llm import generate_response

from vector_store import (
    create_faiss_index,
    load_chunks,
    load_index,
    save_chunks,
    save_index,
    search    
)

import os

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

chunks_storage = []
index_storage = None

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)

    embeddings = get_embedding(chunks)
    index = create_faiss_index(embeddings)

    save_index(index)
    save_chunks(chunks)

    return {"message": "Document processed and saved"}

@app.post("/ask")
async def ask_question(question: str):

    if not os.path.exists("faiss_index.index"):
        return {"error": "Upload document first"}

    index = load_index()
    chunks = load_chunks()

    query_embedding = get_embedding([question])[0]
    top_indices = search(index, query_embedding)

    retrieved_chunks = [chunks[i] for i in top_indices]
    context = "\n\n".join(retrieved_chunks)

    answer = generate_response(context, question)

    return {"answer": answer}
