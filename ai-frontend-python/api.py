from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
import os
import io
from PIL import Image

from smartcs.services import load_vision_model, global_chat, find_visual_matches, analyze_uploaded_document
from smartcs.reporting import FreedomToOperateReport, generate_report_pdf

app = FastAPI(title="SmartCS Neural API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vision_feature_extractor = load_vision_model()

os.makedirs("static", exist_ok=True)
os.makedirs("extracted_images", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/extracted_images", StaticFiles(directory="extracted_images"), name="images")

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open("static/index.html", "r") as f:
        return f.read()

class ChatRequest(BaseModel):
    question: str

@app.post("/api/chat")
def api_chat(req: ChatRequest):
    try:
        result = global_chat(req.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/visual_match")
async def api_visual_match(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        matches = find_visual_matches(image, vision_feature_extractor)
        return {"matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze_document")
async def api_analyze_document(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        filename = file.filename.lower()
        if filename.endswith(".pdf"):
            result = analyze_uploaded_document(contents, is_pdf=True)
        else:
            result = analyze_uploaded_document(contents, is_pdf=False)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/report")
def api_report(report_data: FreedomToOperateReport):
    try:
        pdf_bytes = generate_report_pdf(report_data)
        return Response(content=pdf_bytes, media_type="application/pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
