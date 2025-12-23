"""Minimal FastAPI app demonstrating retrieval + generation using only retrieved context.

Run locally:
    uvicorn app.api:app --host 0.0.0.0 --port 8000

Endpoints:
- POST /query { "question": "...", "top_k": 5 }

Response includes: answer (string) and citations (list of chunk ids + metadata)
"""
import json
import re
from pathlib import Path
from typing import List

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

APP_DIR = Path(__file__).parent
INDEX_DIR = Path("indexes")


class Query(BaseModel):
    question: str
    top_k: int = 5


app = FastAPI(title="RAG demo")

# mount static UI
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


@app.get("/")
def home():
    index_path = APP_DIR / "static" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"status": "UI not found. Run scripts/cache_models.py and start the server."}


def load_index():
    idx_path = INDEX_DIR / "faiss.index"
    meta_path = INDEX_DIR / "metadata.json"
    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Index or metadata not found. Run scripts/build_index.py first.")
    index = faiss.read_index(str(idx_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, metadata


@app.on_event("startup")
def startup():
    global embed_model, index, metadata, gen_tokenizer, gen_model
    embed_model = SentenceTransformer("models/all-MiniLM-L6-v2" if Path("models/all-MiniLM-L6-v2").exists() else "all-MiniLM-L6-v2")
    try:
        index, metadata = load_index()
    except Exception:
        index, metadata = None, None

    gen_tokenizer = AutoTokenizer.from_pretrained("models/flan-t5-base" if Path("models/flan-t5-base").exists() else "google/flan-t5-base")
    gen_model = AutoModelForSeq2SeqLM.from_pretrained("models/flan-t5-base" if Path("models/flan-t5-base").exists() else "google/flan-t5-base")

    # generation / prompt size defaults
    global MAX_MODEL_TOKENS, MAX_NEW_TOKENS, MAX_PROMPT_TOKENS
    try:
        MAX_MODEL_TOKENS = gen_tokenizer.model_max_length if gen_tokenizer.model_max_length is not None else 1024
    except Exception:
        MAX_MODEL_TOKENS = 1024
    MAX_NEW_TOKENS = 256
    # Prompt budget: allow more retrieved text, but not so much that the model is confused.
    MAX_PROMPT_TOKENS = max(450, MAX_MODEL_TOKENS - MAX_NEW_TOKENS)


def retrieve(question: str, top_k: int = 5):
    if index is None:
        raise RuntimeError("Index not loaded. Build the index first.")
    q_emb = embed_model.encode([question])
    q_emb = np.array(q_emb).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        item = metadata[idx]
        hits.append({"id": item["id"], "score": float(score), "text": item["text"], "title": item.get("title"), "url": item.get("url")})
    return hits


def re_rank_hits(hits: List[dict], question: str) -> List[dict]:
    # Simple re-ranker: boost chunks with question keywords
    q_l = question.lower()
    q_words = set(w.lower() for w in re.findall(r"\w+", question))
    
    # Keyword lists for different intents
    symptom_kws = ["symptom", "symptoms", "signs", "present", "blurred vision", "thirsty", "urinate", "tired", "weight loss", "fever", "headache", "chills"]
    prevent_kws = ["prevent", "prevention", "preventable", "avoid", "reduce risk", "lower risk", "vaccination", "vaccine", "immunization", "lifestyle", "diet", "exercise"]
    
    reranked = []
    for h in hits:
        text = (h.get("text") or "").lower()
        title = (h.get("title") or "").lower()
        base = float(h.get("score", 0.0))
        boost = 0.0
        
        # Strong boost for matching keywords
        if any(kw in q_l for kw in prevent_kws):
            for kw in prevent_kws:
                if kw in text or kw in title:
                    boost += 0.8
        
        if any(kw in q_l for kw in symptom_kws):
            for kw in symptom_kws:
                if kw in text or kw in title:
                    boost += 0.7
        
        # Lexical overlap
        hit_words = set(w.lower() for w in re.findall(r"\w+", text))
        overlap = len(q_words & hit_words)
        boost += 0.02 * min(overlap, 20)
        
        h_new = h.copy()
        h_new["score"] = base + boost
        reranked.append(h_new)
    
    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked


def make_prompt(hits: List[dict], question: str) -> str:
    instr = (
        "Answer the question using only the information in the passages below. "
        "Give a complete answer with all the important details and examples mentioned in the passages.\n\n"
    )

    header = instr + "Context passages:\n\n"

    parts = []
    total_tokens = len(gen_tokenizer.encode(header, add_special_tokens=False))
    for i, h in enumerate(hits, start=1):
        chunk_id = h.get("id")
        chunk_text = h.get("text", "")
        piece_prefix = f"[{i}] "
        piece_suffix = f"\n[id:{chunk_id}]\n\n"

        prefix_tokens = len(gen_tokenizer.encode(piece_prefix + piece_suffix, add_special_tokens=False))
        chunk_tokens = gen_tokenizer.encode(chunk_text, add_special_tokens=False)
        avail = MAX_PROMPT_TOKENS - total_tokens - prefix_tokens
        if avail <= 0:
            break
        if len(chunk_tokens) > avail:
            truncated_tokens = chunk_tokens[:avail]
            chunk_snippet = gen_tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        else:
            chunk_snippet = chunk_text

        piece = piece_prefix + chunk_snippet + piece_suffix
        parts.append(piece)
        total_tokens += len(gen_tokenizer.encode(piece, add_special_tokens=False))

    context = "".join(parts)
    prompt = header + context + f"Question: {question}\n\nAnswer concisely with inline citations."
    return prompt


def is_diagnostic_intent(question: str) -> bool:
    pattern = re.compile(r"\b(i have|do i have|could i have|how do i know if i (?:may )?have|should i (?:take|get|be)|what medicine|prescribe|prescription|diagnos|am i infected|do i need (?:treatment|testing))\b", re.I)
    return bool(pattern.search(question))


@app.post("/query")
def query(q: Query):
    try:
        # retrieve a larger candidate set then re-rank
        hits = retrieve(q.question, max(10, q.top_k))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # re-rank and select top-k
    reranked = re_rank_hits(hits, q.question)
    top_hits = reranked[: q.top_k]

    # if user intent looks diagnostic/prescription-like, refuse and return passages
    if is_diagnostic_intent(q.question):
        snippets = []
        for h in top_hits:
            snippet = h.get("text", "")
            if len(snippet) > 800:
                snippet = snippet[:800] + "..."
            snippets.append({"id": h["id"], "title": h.get("title"), "url": h.get("url"), "snippet": snippet, "score": h.get("score")})
        refusal_msg = (
            "I can't provide a medical diagnosis or prescribe treatment. Please consult a qualified healthcare professional. "
            "Below are relevant WHO passages that may help you understand symptoms and testing options."
        )
        return {"answer": refusal_msg, "refusal": True, "citations": snippets}

    # normal generation flow
    prompt = make_prompt(top_hits, q.question)
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_MODEL_TOKENS)
    out = gen_model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, min_length=40, num_beams=2, early_stopping=True)
    answer = gen_tokenizer.decode(out[0], skip_special_tokens=True)

    # Fallback: if the model returns only a chunk id or an extremely short fragment,
    # return the retrieved passages instead so the user gets content rather than an id.
    a_stripped = answer.strip()
    if (a_stripped.startswith("[id:") and a_stripped.endswith("]")) or (len(a_stripped) < 40 and ("chunk_" in a_stripped or "id:" in a_stripped)):
        passages = []
        for h in top_hits:
            text = h.get("text", "")
            passages.append({"id": h["id"], "title": h.get("title"), "url": h.get("url"), "text": text[:1200]})
        fallback = (
            "The model returned only a chunk identifier. Here are the relevant passages from the retrieved documents:\n\n"
            + "\n\n---\n\n".join([f"{p['id']} - {p['title']}\n{p['text']}" for p in passages])
        )
        return {"answer": fallback, "citations": top_hits, "fallback_used": True}

    return {"answer": answer, "citations": top_hits}
