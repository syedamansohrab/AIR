# ⚖️ SmartCS — Cloud-Scale Patent Intelligence Engine

A multimodal, enterprise-grade AI platform for **Patent Prior Art Search**, **Visual Structural Plagiarism Detection**, **Freedom to Operate (FTO) Analysis**, and **Automated Legal Report Generation**.

Built on a **Hybrid RAG Architecture** — it combines a proprietary Pinecone vector database with Google Gemini's global knowledge to answer patent questions on **any topic in the world**.

---

## 🧠 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SmartCS Intelligence Engine                   │
├─────────────────────┬───────────────────────────────────────────┤
│   Java Backend      │   Python AI Engine (FastAPI)              │
│   (Apache Lucene)   │                                           │
│                     │  ┌──────────────┐  ┌──────────────────┐   │
│  • BM25 Text Search │  │  Pinecone    │  │  Google Gemini   │   │
│  • TF-IDF Scoring   │  │  Vector DB   │  │  (Embeddings +   │   │
│  • REST API :8080   │  │  (3072-dim)  │  │   Generation)    │   │
│                     │  └──────┬───────┘  └────────┬─────────┘   │
│                     │         │    Hybrid RAG      │             │
│                     │         └────────┬───────────┘             │
│                     │                  │                         │
│                     │  ┌───────────────▼──────────────────────┐  │
│                     │  │  PyTorch ResNet50 (Vision Engine)    │  │
│                     │  │  FPDF2 (PDF Report Generator)       │  │
│                     │  └─────────────────────────────────────┘  │
├─────────────────────┴───────────────────────────────────────────┤
│                   Dark-Mode Glassmorphic UI                      │
│               http://localhost:8000                              │
└─────────────────────────────────────────────────────────────────┘
```

### Core Modules

| Module | Technology | Description |
|--------|-----------|-------------|
| **Semantic Search** | Pinecone + Gemini Embeddings | Cloud vector similarity search across patent chunks (3072-dim) |
| **Hybrid RAG** | Gemini Flash + Smart Thresholds | Blends local DB results with Gemini's global knowledge based on relevance scores |
| **Visual Detection** | PyTorch ResNet50 | Converts engineering diagrams to feature vectors for structural similarity matching |
| **FTO Analysis** | Gemini Flash | Uploads a PDF proposal and cross-references it against all prior art in the database |
| **Report Generator** | FPDF2 | One-click PDF generation aggregating all chat + visual analysis from the session |
| **Text Search** | Apache Lucene (Java) | BM25/TF-IDF keyword-based search as a supplementary retrieval layer |

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **AI Engine** | Python 3.12, FastAPI, Uvicorn |
| **LLM / Embeddings** | Google Gemini Flash, Gemini Embedding 001 |
| **Vector Database** | Pinecone (Serverless, 3072-dim, Cosine) |
| **Computer Vision** | PyTorch, TorchVision, ResNet50 |
| **Search Backend** | Java 17+, Maven, Apache Lucene |
| **PDF Processing** | PyMuPDF (Fitz), FPDF2 |
| **Frontend** | HTML5, CSS3, Vanilla JS, Marked.js |
| **Containerization** | Docker, Docker Compose |

---

## 📋 Prerequisites

Before running this project, ensure you have:

- **Python 3.10+** (with `pip` and `venv`)
- **Java 11+** and **Maven** (for the Lucene backend)
- **Git**
- **API Keys** (both free-tier):
  - 🔑 **Google Gemini API Key** — get yours at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
  - 🔑 **Pinecone API Key** — get yours at [app.pinecone.io](https://app.pinecone.io)

---

## 🚀 Quick Start (Step-by-Step)

> You will need **two terminal windows** open simultaneously.

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/Patent-IR-Engine.git
cd Patent-IR-Engine
```

### Step 2: Configure API Keys

```bash
# Copy the example environment file
cp ai-frontend-python/.env.example ai-frontend-python/.env

# Open and paste your real API keys
nano ai-frontend-python/.env
```

Your `.env` should look like:
```
GEMINI_API_KEY=AIzaSy...your_real_key
PINECONE_API_KEY=pcsk_...your_real_key
```

### Step 3: Start the Java Backend (Terminal 1)

```bash
cd backend-java
mvn clean compile exec:java -Dexec.mainClass="com.smartcs.indexer.Main"
```

> ⚠️ Wait until you see: `API is LIVE on http://localhost:8080` before continuing.

### Step 4: Setup the Python AI Engine (Terminal 2)

```bash
cd ai-frontend-python

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate          # Windows

# Install all dependencies (from root-level requirements.txt)
pip install -r ../requirements.txt
```

### Step 5: Extract Visuals & Build the Vision Index

These scripts extract diagrams from the PDFs in `data-corpus/` and pass them through the ResNet50 CNN to generate the visual similarity index.

```bash
# Extract images from patent PDFs
python extract_images.py

# Build the PyTorch feature vector index (takes ~30 seconds)
python build_visual_index.py
```

### Step 6: Ingest Patents into Pinecone (One-Time Setup)

This uploads your local patent PDFs into the cloud Pinecone vector database so the RAG engine can search them.

```bash
python ingest_to_pinecone.py
```

> ℹ️ This only needs to be run once. The vectors persist in Pinecone's cloud.

### Step 7: Launch the AI Engine

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Step 8: Open the UI

Navigate to **[http://localhost:8000](http://localhost:8000)** in your browser. 🎉

---

## 🐳 Docker Deployment (Alternative)

If you prefer containerized deployment:

```bash
# From the project root (Patent-IR-Engine/)
docker compose up --build
```

This launches both the Java backend (`:8080`) and Python AI engine (`:8000`) as coordinated containers.

> **Note:** You must have `ai-frontend-python/.env` configured with your API keys before building.

---

## 💡 How to Use

### 1. Chat (Prior Art Search)
Type any patent-related question in the search bar and press Enter. The system will:
- Search your Pinecone vector database for relevant patent chunks
- If highly relevant matches exist (score ≥ 0.82), it uses your local patent corpus
- If the topic is outside your corpus, it seamlessly switches to Gemini's global knowledge

### 2. Upload a Document (FTO Analysis)
Click the **📎 paperclip icon** → select a **PDF** or **text file** containing an invention proposal. The engine will:
- Extract text from the document
- Embed it and search for overlapping prior art
- Generate a comparative Freedom to Operate report

### 3. Visual Structural Check
Click the **📎 paperclip icon** → select an **image file** (PNG/JPG of an engineering diagram). The engine will:
- Pass the image through the ResNet50 CNN
- Find the most structurally similar patent diagrams in the database

### 4. Download Report
After performing any analysis, click the **📄 report icon** in the input bar. A professional PDF report aggregating all your session's analyses will be generated and downloaded instantly.

---

## 📁 Project Structure

```
Patent-IR-Engine/
├── README.md                          # This file
├── Dockerfile.java                    # Docker instructions for Java
├── Dockerfile.python                  # Docker instructions for Python
├── docker-compose.yml                 # Multi-container orchestration
├── .dockerignore                      # Docker exclusion rules
├── .gitignore
│
├── data-corpus/                       # Patent PDFs (15 sample patents included)
│   ├── EP3526119B1.pdf
│   ├── US10058149.pdf
│   └── ... (15 total)
│
├── backend-java/                      # Java Lucene Search Backend
│   ├── pom.xml
│   └── src/
│
└── ai-frontend-python/               # Python AI Engine
    ├── .env.example                   # API key template
    ├── .env                           # Your actual keys (git-ignored)
    ├── api.py                         # FastAPI routes
    ├── ingest_to_pinecone.py          # One-time Pinecone data loader
    ├── extract_images.py              # PDF image extraction script
    ├── build_visual_index.py          # ResNet50 visual indexer
    │
    ├── smartcs/                       # Core AI module
    │   ├── __init__.py
    │   ├── config.py                  # Paths and settings
    │   ├── services.py                # RAG, FTO, Vision logic
    │   └── reporting.py               # PDF report generator
    │
    ├── static/                        # Frontend assets
    │   ├── index.html
    │   ├── style.css
    │   └── script.js
    │
    └── extracted_images/              # Auto-generated patent diagrams
```

---

## ⚙️ Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ | Google Gemini API key for embeddings + generation |
| `PINECONE_API_KEY` | ✅ | Pinecone vector database API key |

---

## 🧪 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Make sure your `.venv` is activated: `source .venv/bin/activate` |
| `uvicorn: command not found` | Run via the venv directly: `.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000` |
| Report button does nothing | Make sure you've performed at least one chat/analysis before clicking the report button |
| Unicode PDF crash | The `_safe_text()` function in `reporting.py` handles this. If you still see issues, ensure `fpdf2>=2.7.9` is installed |
| Stale UI after code changes | Hard-refresh your browser: `Ctrl+Shift+R` (clears cached JS/CSS) |

---

## 📜 License

This project is licensed under the **Apache License 2.0**.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

> Built with ❤️ using PyTorch, Google Gemini, Pinecone, Apache Lucene, and FastAPI.