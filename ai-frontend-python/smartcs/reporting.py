import datetime
from pydantic import BaseModel
from typing import List

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

class NLPExtractObj(BaseModel):
    target: str
    inquiry: str
    exact_answer: str

class StructuralMatchObj(BaseModel):
    source_patent: str
    similarity: float
    image_filename: str

class FreedomToOperateReport(BaseModel):
    nlp_extracts: List[NLPExtractObj] = []
    structural_matches: List[StructuralMatchObj] = []

def _safe_text(value: str) -> str:
    """Safely cast markdown emojis/unicode to latin-1 to prevent fpdf crashing"""
    if not value: return ""
    return str(value).encode('latin-1', 'replace').decode('latin-1')

if FPDF is not None:
    class SmartCSPdfReport(FPDF):
        def header(self):
            if self.page_no() > 1:
                self.set_fill_color(30, 44, 58)
                self.rect(0, 0, 210, 20, style="F")
                self.set_text_color(255, 255, 255)
                self.set_font("Helvetica", "B", 12)
                self.set_y(6)
                self.cell(0, 8, "SmartCS FTO Architecture Report", align="L")
                self.set_font("Helvetica", "I", 10)
                self.cell(0, 8, datetime.datetime.now().strftime("%Y-%m-%d"), align="R")
                self.ln(15)

        def footer(self):
            if self.page_no() > 1:
                self.set_y(-15)
                self.set_text_color(150, 150, 150)
                self.set_font("Helvetica", "I", 8)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")

        def title_page(self):
            self.add_page()
            self.set_fill_color(18, 25, 38)
            self.rect(0, 0, 210, 297, style="F")
            self.set_y(100)
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 36)
            self.cell(0, 15, "SmartCS", align="C", new_x="LMARGIN", new_y="NEXT")
            self.set_font("Helvetica", "", 18)
            self.set_text_color(94, 106, 210)
            self.cell(0, 15, "Freedom To Operate Report", align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(20)
            self.set_font("Helvetica", "I", 12)
            self.set_text_color(150, 160, 180)
            self.cell(0, 10, f"Generated automatically on {datetime.datetime.now().strftime('%B %d, %Y')}", align="C", new_x="LMARGIN", new_y="NEXT")

        def section_title(self, title: str):
            self.ln(5)
            self.set_font("Helvetica", "B", 14)
            self.cell(0, 10, _safe_text(title), fill=False)
            self.ln(10)


def generate_report_pdf(report: FreedomToOperateReport) -> bytes:
    if FPDF is None:
        raise RuntimeError("PDF generation requires 'fpdf2'.")

    pdf = SmartCSPdfReport()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.title_page()
    pdf.add_page()
    pdf.set_text_color(40, 40, 40)
    
    pdf.section_title("I. NLP FTO Cross-Referencing")
    if not report.nlp_extracts:
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, "No document or chat analyses were logged.")
        pdf.ln(5)
    else:
        for idx, extract in enumerate(report.nlp_extracts):
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 6, _safe_text(f"Event {idx+1}: {extract.inquiry}"))
            pdf.set_x(10) # Reset X explicitly
            pdf.set_font("Helvetica", "I", 10)
            pdf.set_text_color(100, 100, 100)
            pdf.multi_cell(0, 6, _safe_text(f"Sources Used: {extract.target}"))
            pdf.set_x(10)
            
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)
            pdf.multi_cell(0, 6, _safe_text(extract.exact_answer))
            pdf.set_x(10)
            pdf.ln(5)

    pdf.section_title("II. Visual Structural Check")
    if not report.structural_matches:
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, "No structural visual matches were made.")
        pdf.ln(5)
    else:
        for idx, match in enumerate(report.structural_matches):
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, _safe_text(f"{idx+1}. Patent: {match.source_patent}"))
            pdf.ln(8)
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(94, 106, 210)
            pdf.cell(0, 6, _safe_text(f"Geometric Similarity Match score: {match.similarity:.2%}"))
            pdf.ln(8)
            pdf.set_text_color(40, 40, 40)
            pdf.ln(2)

    return bytes(pdf.output())
