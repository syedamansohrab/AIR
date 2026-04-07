import streamlit as st
import requests
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import fitz  # PyMuPDF
from PIL import Image
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="SmartCS Patent & Legal Search", layout="wide")
st.title("⚖️ SmartCS Patent Intelligence Engine")
st.write("Enterprise-grade multimodal search for prior art, visual infringement, and legal extraction.")

# --- AI & DATA LOADING (CACHED FOR SPEED) ---
@st.cache_resource
def load_vision_model():
    resnet = models.resnet50(pretrained=True)
    feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    feature_extractor.eval()
    return feature_extractor

@st.cache_data
def load_visual_index():
    try:
        with open("visual_index.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_resource
def load_qa_model():
    # Reverting to the raw PyTorch model to bypass the broken HuggingFace pipeline bug
    model_name = "deepset/minilm-uncased-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model

feature_extractor = load_vision_model()
visual_index = load_visual_index()
qa_pipeline = load_qa_model() # Returns (tokenizer, model)

# Image Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- UI TABS ---
tab1, tab2, tab3 = st.tabs([
    "📝 Text Search (Units 1 & 2)", 
    "🖼️ Visual Plagiarism Check (Unit 3)", 
    "🧠 Neural Document Q&A (Unit 4)"
])

# ==========================================
# TAB 1: TEXT SEARCH (Lucene BM25)
# ==========================================
with tab1:
    st.subheader("Keyword & Prior Art Search")
    query = st.text_input("Enter patent keywords (e.g., 'drone folding mechanism'):")
    
    if st.button("Search Text"):
        if query:
            try:
                response = requests.get(f"http://localhost:8080/search?q={query}")
                results = response.json()
                
                if results:
                    st.success(f"Found {len(results)} matching patents!")
                    for res in results:
                        st.markdown(f"**📄 {res['filename']}**")
                        st.caption(f"Relevance Score (BM25): {res['score']}")
                        st.divider()
                else:
                    st.warning("No matching patents found.")
            except Exception as e:
                st.error("Backend Java Server is not running! Please start it on port 8080.")

# ==========================================
# TAB 2: IMAGE SEARCH (ResNet50 + Cosine Similarity)
# ==========================================
with tab2:
    st.subheader("Visual Patent Infringement Check")
    st.write("Upload a diagram to find visually similar mechanical designs.")
    
    if visual_index is None:
        st.error("Visual Index not found! Please run `python build_visual_index.py` first.")
    else:
        uploaded_file = st.file_uploader("Upload engineering diagram", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption="Uploaded Diagram", width=300)
            
            if st.button("Run AI Visual Match"):
                with st.spinner("Analyzing geometric features via ResNet50..."):
                    img_t = preprocess(img)
                    batch_t = torch.unsqueeze(img_t, 0)
                    
                    with torch.no_grad():
                        query_features = feature_extractor(batch_t).numpy().flatten().reshape(1, -1)
                    
                    db_filenames = list(visual_index.keys())
                    db_features = np.array(list(visual_index.values()))
                    
                    similarities = cosine_similarity(query_features, db_features)[0]
                    top_indices = similarities.argsort()[-3:][::-1]
                    
                    st.success("Analysis Complete! Top visual matches:")
                    cols = st.columns(3)
                    for i, idx in enumerate(top_indices):
                        match_filename = db_filenames[idx]
                        match_score = similarities[idx]
                        
                        with cols[i]:
                            st.image(f"extracted_images/{match_filename}", use_container_width=True)
                            st.caption(f"**Origin:** {match_filename.split('_page')[0]}.pdf")
                            st.caption(f"**Similarity:** {match_score:.2%}")

# ==========================================
# TAB 3: NEURAL Q&A (HuggingFace Transformers)
# ==========================================
with tab3:
    st.subheader("Deep Document Reading (AI Extraction)")
    st.write("Select a patent and ask the AI a specific question about its contents.")
    
    corpus_dir = "../data-corpus"
    if os.path.exists(corpus_dir):
        patents = [f for f in os.listdir(corpus_dir) if f.endswith(".pdf")]
        
        selected_patent = st.selectbox("1. Select Patent to Analyze:", patents)
        user_question = st.text_input("2. Ask a precise question (e.g., 'What comprises the folding arm?'):")
        
        if st.button("Extract Answer"):
            if selected_patent and user_question:
                with st.spinner("AI is reading the document using a sliding window..."):
                    pdf_path = os.path.join(corpus_dir, selected_patent)
                    doc = fitz.open(pdf_path)
                    
                    # Grab first 5 pages
                    full_text = ""
                    for page_num in range(min(5, len(doc))):
                        full_text += doc.load_page(page_num).get_text() + " "
                        
                    if len(full_text.strip()) > 0:
                        try:
                            tokenizer, qa_model = qa_pipeline
                            
                            # --- CUSTOM SLIDING WINDOW ALGORITHM ---
                            words = full_text.split()
                            chunk_size = 350
                            overlap = 50
                            
                            best_score = float('-inf')
                            best_answer = ""
                            
                            # Scan through the document in chunks
                            for i in range(0, len(words), chunk_size - overlap):
                                chunk_text = " ".join(words[i:i+chunk_size])
                                
                                inputs = tokenizer(user_question, chunk_text, return_tensors="pt", truncation=True, max_length=512)
                                
                                with torch.no_grad():
                                    outputs = qa_model(**inputs)
                                    
                                start_idx = outputs.start_logits.argmax()
                                end_idx = outputs.end_logits.argmax()
                                
                                # Only consider valid answers (start index > 0 ignores the [CLS] empty token)
                                if start_idx > 0 and start_idx <= end_idx:
                                    # Calculate confidence score based on raw logits
                                    score = outputs.start_logits[0, start_idx].item() + outputs.end_logits[0, end_idx].item()
                                    
                                    if score > best_score:
                                        answer_tokens = inputs.input_ids[0, start_idx : end_idx + 1]
                                        temp_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
                                        
                                        if len(temp_answer.strip()) > 2:
                                            best_score = score
                                            best_answer = temp_answer
                            # ----------------------------------------
                            
                            if best_answer:
                                st.success("Answer Extracted via Neural Tensors!")
                                st.markdown(f"### 🤖 **Answer:** {best_answer}")
                                
                                snippet_idx = full_text.lower().find(best_answer.lower())
                                if snippet_idx != -1:
                                    start_snip = max(0, snippet_idx - 60)
                                    end_snip = snippet_idx + len(best_answer) + 60
                                    st.info(f"**Context Snippet found in PDF:**\n\n...{full_text[start_snip:end_snip]}...")
                            else:
                                st.warning("The AI could not find a highly confident answer in the text.")
                                
                        except Exception as e:
                            st.error(f"AI could not extract an answer: {e}")
                    else:
                        st.warning("Could not extract readable text from this PDF.")
    else:
        st.error("Data corpus folder not found!")