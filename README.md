# ⚖️ SmartCS Patent Intelligence Engine

An enterprise-grade, multimodal microservice architecture designed to automate patent prior art retrieval, detect visual mechanical infringement, and perform deep legal document reading comprehension.

---

## 🚀 System Architecture & Features

This engine is divided into three core AI and Search modules:
* **Units 1 & 2: Semantic Prior Art Search (Java Backend)**
    * Custom Information Retrieval system built with **Apache Lucene**.
    * Executes BM25 (TF-IDF) mathematical scoring to rank document relevance based on text density.
* **Unit 3: Visual Plagiarism Detection (Computer Vision)**
    * Utilizes a **PyTorch ResNet50** Convolutional Neural Network (CNN).
    * Converts mechanical engineering diagrams into high-dimensional feature vectors to detect structural patent infringement via **Cosine Similarity**.
* **Unit 4: Neural Document Q&A (NLP)**
    * Implements a local HuggingFace Transformer (`MiniLM-QA`) via PyTorch tensors.
    * Uses a custom **Sliding Window Context Algorithm** to bypass token limits, allowing the AI to read deeply through European/US patent disclaimers and extract precise engineering definitions.

---

## 🛠️ Tech Stack
* **Frontend & AI Inference:** Python 3.12, Streamlit, PyTorch, HuggingFace Transformers, Scikit-Learn
* **Search Backend:** Java 11+, Maven, Apache Lucene
* **Data Processing:** PyMuPDF (Fitz), Pillow

---

## 📋 Prerequisites

Before running this project, ensure you have the following installed on your system:
* **Java 11 or higher**
* **Maven**
* **Python 3.10 or higher** (with `pip` and `venv`)
* **Git**

---

## 💻 Step-by-Step Setup & Execution

This application runs as a microservice architecture. You will need **two separate terminal windows** open to run the backend and frontend simultaneously.

### Step 1: Prepare the Data Corpus
Ensure you have a folder named `data-corpus` in the root directory containing your PDF patents (English language `US` or `EP` patents are recommended).

### Step 2: Start the Java Backend (Terminal 1)
This initiates the Lucene indexing process and opens the REST API on port 8080.


# Navigate to the backend folder
cd backend-java

# Compile the Java code and run the Lucene server
mvn clean compile exec:java -Dexec.mainClass="com.smartcs.indexer.Main"

(⚠️ Do not proceed to Step 3 until you see the message: API is LIVE on http://localhost:8080)

###Step 3: Setup the Python AI Environment (Terminal 2)
Open a new terminal window to configure the local AI environment.

# Navigate to the frontend folder
cd ai-frontend-python

# Create a fresh virtual environment
python3 -m venv .venv

# Activate the virtual environment
# (On Windows use: .venv\Scripts\activate)
source .venv/bin/activate

# Install all required AI and UI libraries
pip install -r requirements.txt

### Step 4: Extract Images & Build the Visual AI Index
Before searching, you must extract the diagrams from the PDFs and pass them through the ResNet50 model to create the mathematical visual index.

# 1. Extract diagrams from the PDFs
python extract_images.py

# 2. Convert images to PyTorch Tensors (This may take a moment)
python build_visual_index.py

### Step 5: Launch the UI
Once the visual index is built, launch the Streamlit web application.

streamlit run app.py

The application will automatically open in your default web browser at http://localhost:8501.
