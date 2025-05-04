# app/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag import index_document, retrieve

app = FastAPI(title="RAG Chatbot API")

class IndexRequest(BaseModel):
    text: str
    doc_id: int

class RetrieveRequest(BaseModel):
    query: str
    k: int = 3

@app.post("/index", summary="Index a new document")
def api_index(req: IndexRequest):
    try:
        index_document(req.text, req.doc_id)
        return {"status": "ok", "doc_id": req.doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve", summary="Retrieve top-k document IDs")
def api_retrieve(req: RetrieveRequest):
    try:
        ids = retrieve(req.query, k=req.k)
        return {"status": "ok", "doc_ids": ids.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
