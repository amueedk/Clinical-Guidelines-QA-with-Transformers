# Clinical Guidelines QA with Transformers + RAG (Patient-Safe Assistant)

**Evidence-grounded clinical Q&A system** using Retrieval-Augmented Generation (RAG) to answer health questions from WHO public health guidelines. Built with small transformer models (`flan-t5-base`) optimized for CPU inference.

## ✨ Features

- 🔍 **FAISS-based retrieval** with semantic search over chunked WHO documents
- 🤖 **Small LLM generation** using Google's FLAN-T5-base (CPU-friendly)
- 📊 **Re-ranking** with intent-aware keyword boosting for better accuracy
- 🔗 **Chunk-level citations** tracking source documents and URLs
- ⚠️ **Medical safety** disclaimers and diagnostic intent detection
- 📈 **Evaluation framework** with Recall@k and answer quality metrics
- 🌐 **Interactive web UI** built with FastAPI

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB+ RAM (8GB recommended for better performance)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd "gen ai phase 2"
```

2. **Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare corpus and build index**
```bash
# Parse and chunk WHO documents
python scripts/prepare_corpus.py

# Build FAISS index (downloads embedding model)
python scripts/build_index.py
```

5. **Start the server**
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

6. **Open in browser**
Navigate to: `http://localhost:8000`

## 📁 Project Structure

```
.
├── data/
│   └── raw/who/              # WHO text documents (included)
├── scripts/
│   ├── prepare_corpus.py     # Parse and chunk documents
│   ├── build_index.py        # Build FAISS index
│   └── cache_models.py       # Pre-download models (optional)
├── app/
│   ├── api.py                # FastAPI RAG endpoint
│   └── static/index.html     # Web UI
├── eval/
│   ├── test_questions.json   # Labeled test set (Recall@k)
│   ├── answer_questions.json # Answer evaluation dataset (32 Q/A)
│   ├── retrieval_eval.py     # Recall@k evaluation script
│   └── answer_eval.py        # Answer quality evaluation
└── requirements.txt
```

## 📊 Evaluation

Run retrieval and answer quality evaluations:

```bash
# Start server first
uvicorn app.api:app

# In new terminal:
# Evaluate retrieval (Recall@5)
python eval/retrieval_eval.py --k 5

# Evaluate answer quality
python eval/answer_eval.py --top_k 8
```

Results saved to `eval/retrieval_results.json` and `eval/answer_results.json`.

## 🔧 Configuration

**Chunking** (`scripts/prepare_corpus.py`):
- Token-based chunking with `chunk_size=300`, `overlap=60`

**Retrieval** (`app/api.py`):
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Index: FAISS with L2 distance
- Re-ranking: Intent-aware keyword boosting

**Generation** (`app/api.py`):
- Model: `google/flan-t5-base`
- Max new tokens: 256
- Min length: 40
- Beam search: 2 beams

## 🎯 Use Cases

**✅ Appropriate:**
- General health information queries
- Prevention and lifestyle guidance
- Understanding diseases and symptoms
- Vaccination information

**❌ Not appropriate:**
- Personal medical diagnosis
- Prescription recommendations
- Emergency medical advice
- Replacing professional healthcare

## 📝 Example Queries

- "What are the symptoms of diabetes?"
- "How can malaria be prevented?"
- "What causes hepatitis A?"
- "What are the health consequences of obesity?"

## ⚠️ Disclaimer

This system is for **educational purposes only**. It provides general health information from WHO guidelines and is **not a substitute for professional medical advice, diagnosis, or treatment**. Always consult qualified healthcare providers for medical concerns.

## 🛠️ Technical Details

- **Embedding**: 384-dim vectors from MiniLM-L6-v2
- **Index**: FAISS Flat (exact search)
- **Chunking**: 300 tokens with 60-token overlap
- **Prompt budget**: 450 tokens max
- **Generation**: CPU-optimized with beam search
- **Re-ranking**: Symptom/prevention keyword boosting

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

**Built with:** Transformers • FastAPI • FAISS • Sentence Transformers
