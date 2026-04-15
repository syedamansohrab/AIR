import os
import io
import google.generativeai as genai
from pinecone import Pinecone
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from dotenv import load_dotenv

from smartcs.config import EXTRACTED_IMAGES_DIR

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

pc = None
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)

def load_vision_model():
    resnet = models.resnet50(pretrained=True)
    feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    feature_extractor.eval()
    return feature_extractor

def build_image_preprocess():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def find_visual_matches(uploaded_image: Image.Image, feature_extractor, top_k: int = 3):
    if not pc: raise Exception("Pinecone not configured")
    index = pc.Index("smartcs-patents-vision")
    
    preprocess = build_image_preprocess()
    img_tensor = preprocess(uploaded_image)
    batch_tensor = torch.unsqueeze(img_tensor, 0)

    with torch.no_grad():
        query_features = feature_extractor(batch_tensor).numpy().flatten().tolist()
        
    response = index.query(
        vector=query_features,
        top_k=top_k,
        include_metadata=True
    )
    
    matches = []
    for match in response['matches']:
        matches.append({
            "image_filename": match['id'],
            "source_patent": match['metadata'].get('source_patent', 'Unknown'),
            "similarity": match['score'],
            "image_url": f"/extracted_images/{match['id']}"
        })
    return matches

def analyze_uploaded_document(file_bytes: bytes, is_pdf: bool = True):
    if not pc or not GEMINI_API_KEY:
        raise Exception("Pinecone or Gemini API Keys not configured.")
        
    if is_pdf:
        import fitz
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        full_text = ""
        for page_num in range(min(5, len(pdf_document))):
            page = pdf_document[page_num]
            full_text += page.get_text()
    else:
        full_text = file_bytes.decode("utf-8")
        
    if len(full_text) > 10000:
        full_text = full_text[:10000]
        
    index = pc.Index("smartcs-patents-text")
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=full_text,
        task_type="retrieval_query",
    )
    query_vector = result['embedding']
    
    response = index.query(
        vector=query_vector,
        top_k=5,
        include_metadata=True
    )
    
    sources = set()
    context_chunks = []
    for match in response['matches']:
        sources.add(match['metadata']['source_patent'])
        context_chunks.append(f"[{match['metadata']['source_patent']}] {match['metadata']['text']}")
        
    prior_art_context = "\n\n---\n\n".join(context_chunks)
    
    model = genai.GenerativeModel('gemini-flash-latest')
    prompt = f"""
    You are an expert Patent Examiner. A company has submitted a new invention proposal document.
    Your task is to perform a Freedom to Operate (FTO) Analysis by comparing the Proposal Document against the existing Prior Art retrieved from our global database.
    
    Structure your report:
    1. Overall Verdict (Is the work already done?)
    2. Overlaps with Prior Art
    3. Potential Novelty (Differences)
    4. Relevant Patents to Review
    
    --- UPLOADED PROPOSAL DOCUMENT (Extract) ---
    {full_text}
    
    --- EXISTING PRIOR ART DATABASE MATCHES ---
    {prior_art_context}
    """
    
    answer_res = model.generate_content(prompt)
    
    return {
        "analysis": answer_res.text,
        "closest_patents": list(sources)
    }

def global_chat(question: str):
    if not GEMINI_API_KEY:
        raise Exception("Gemini API Key not configured.")
    
    RELEVANCE_THRESHOLD = 0.82  # Only truly on-topic matches pass through
    
    # --- Phase 1: Check local Pinecone vector DB ---
    local_sources = set()
    local_context_chunks = []
    pinecone_available = pc is not None
    max_score = 0.0
    
    if pinecone_available:
        try:
            index = pc.Index("smartcs-patents-text")
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=question,
                task_type="retrieval_query",
            )
            query_vector = result['embedding']
            
            response = index.query(
                vector=query_vector,
                top_k=10,
                include_metadata=True
            )
            
            for match in response['matches']:
                score = match.get('score', 0)
                if score >= RELEVANCE_THRESHOLD:
                    local_sources.add(match['metadata']['source_patent'])
                    local_context_chunks.append(
                        f"[{match['metadata']['source_patent']}] (relevance: {score:.2f}) {match['metadata']['text']}"
                    )
                if score > max_score:
                    max_score = score
        except Exception:
            pinecone_available = False
    
    # --- Phase 2: Determine strategy ---
    has_local_hits = len(local_context_chunks) > 0
    local_context = "\n\n---\n\n".join(local_context_chunks)
    
    model = genai.GenerativeModel('gemini-flash-latest')
    
    if has_local_hits and max_score >= 0.88:
        # HIGH RELEVANCE: Local DB has strong matches — use them primarily
        prompt = f"""You are SmartCS, an expert Patent Intelligence Engine with access to a proprietary patent vector database.
        
Use the following retrieved patent excerpts to provide a comprehensive, expert-level answer. 
Cite every patent source used in brackets, e.g. [US12345A1.pdf].
If the retrieved context is insufficient to fully answer, supplement with your own global patent knowledge and clearly mark those sections as [Global AI Knowledge].

--- RETRIEVED FROM SMARTCS DATABASE (High Confidence) ---
{local_context}

User Question: {question}"""
        sources_list = list(local_sources)
        
    elif has_local_hits and max_score >= RELEVANCE_THRESHOLD:
        # MEDIUM RELEVANCE: Blend local DB + global knowledge
        prompt = f"""You are SmartCS, an expert Patent Intelligence Engine. 
        
The user's question partially relates to patents in our database. Use the retrieved excerpts below as supporting context, but also draw heavily from your comprehensive global knowledge of patents, engineering, and technology to provide a complete answer.
Cite local sources in brackets e.g. [US12345A1.pdf]. Mark information from your global knowledge as [Global AI Knowledge].

--- RETRIEVED FROM SMARTCS DATABASE (Partial Match) ---
{local_context}

User Question: {question}"""
        sources_list = list(local_sources) + ["Global AI Knowledge"]
        
    else:
        # LOW/NO RELEVANCE: Topic is outside our 15 local PDFs — go fully global
        prompt = f"""You are SmartCS, an expert Patent Intelligence Engine with access to the world's largest AI knowledge base.

The user is asking about a topic that is NOT covered in our local proprietary database. 
Use your vast training knowledge (which includes millions of patents, research papers, and engineering documents from across the internet) to provide a highly detailed, expert-level patent intelligence answer.

When referencing real patents you know about, cite them by their actual patent numbers (e.g., US1234567, EP1234567, WO2024123456).
Structure your answer professionally with clear sections and headers.

User Question: {question}"""
        sources_list = ["Global AI Knowledge"]
    
    answer_res = model.generate_content(prompt)
    
    return {
        "answer": answer_res.text,
        "sources": sources_list
    }
