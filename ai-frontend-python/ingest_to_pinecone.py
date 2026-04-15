import os
import pickle
import fitz
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    print("Please configure .env with PINECONE_API_KEY and GEMINI_API_KEY")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

TEXT_INDEX_NAME = "smartcs-patents-text"
VISION_INDEX_NAME = "smartcs-patents-vision"
CORPUS_DIR = "../data-corpus"
VISUAL_INDEX_FILE = "visual_index.pkl"

def ingest_visual_vectors():
    print(f"🔄 Uploading visual vectors to Pinecone ({VISION_INDEX_NAME})...")
    index = pc.Index(VISION_INDEX_NAME)
    
    if not os.path.exists(VISUAL_INDEX_FILE):
        print(f"Skipping visual ingestion. {VISUAL_INDEX_FILE} not found.")
        return
        
    with open(VISUAL_INDEX_FILE, "rb") as f:
        visual_index = pickle.load(f)
        
    vectors = []
    for filename, vector in visual_index.items():
        doc_id = filename
        vectors.append({
            "id": doc_id, 
            "values": vector.tolist(), 
            "metadata": {"source_patent": filename.split('_page')[0] + '.pdf'}
        })
        
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i+batch_size])
    print(f"✅ Uploaded {len(vectors)} visual vectors to cloud!")

def chunk_text(text, chunk_size=1500, overlap=150):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size].strip())
    return [c for c in chunks if len(c) > 50]

def ingest_text_vectors():
    print(f"🔄 Uploading NLP text vectors to Pinecone ({TEXT_INDEX_NAME})...")
    index = pc.Index(TEXT_INDEX_NAME)
    
    patents = [f for f in os.listdir(CORPUS_DIR) if f.endswith(".pdf")]
    
    for patent in patents:
        print(f"  ... processing {patent}")
        doc = fitz.open(os.path.join(CORPUS_DIR, patent))
        text = ""
        for page in doc:
            text += page.get_text()
            
        chunks = chunk_text(text)
        vectors = []
        for i, chunk in enumerate(chunks):
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=chunk,
                task_type="retrieval_document",
            )
            embedding = result['embedding']
            
            vectors.append({
                "id": f"{patent}-chunk-{i}",
                "values": embedding,
                "metadata": {"source_patent": patent, "text": chunk}
            })
            
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            index.upsert(vectors=vectors[i:i+batch_size])
            
    print("✅ Uploaded all text abstracts & claims to cloud!")

if __name__ == "__main__":
    print("--- STARTING VECTOR DB MIGRATION ---")
    ingest_visual_vectors()
    ingest_text_vectors()
    print("--- MIGRATION COMPLETE! ---")
