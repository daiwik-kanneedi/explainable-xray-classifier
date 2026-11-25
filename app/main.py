"""FastAPI entrypoint: GET / (upload form), POST /predict (HTMX partial)."""
from __future__ import annotations

import base64
import io
from datetime import datetime

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.config import BASE_DIR, MODEL_PATH
from app.ml.inference import Predictor

app = FastAPI(title="Explainable Chest X-Ray")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html", {"request": request, "model_ready": MODEL_PATH.exists()}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    if not MODEL_PATH.exists():
        return templates.TemplateResponse(
            "_result.html",
            {"request": request, "error": "No trained model. Run: python -m scripts.train"},
        )
    try:
        result = Predictor().predict(await file.read())
    except Exception as exc:
        return templates.TemplateResponse(
            "_result.html", {"request": request, "error": f"Prediction failed: {exc}"}
        )
    return templates.TemplateResponse("_result.html", {"request": request, **result})


def _base64_to_image(b64_str: str) -> io.BytesIO:
    """Helper to convert base64 string from inference to a file-like object for reportlab."""
    img_data = b64_str.split(",")[1]
    return io.BytesIO(base64.b64decode(img_data))


@app.post("/download_report")
async def download_report(request: Request, file: UploadFile = File(...)):
    result = Predictor().predict(await file.read())
    
    # Create PDF in memory
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer, pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'], fontSize=20, spaceAfter=20, textColor=colors.HexColor("#1e3a8a")
    )
    heading_style = ParagraphStyle(
        'CustomHeading', parent=styles['Heading2'], fontSize=14, spaceBefore=15, spaceAfter=10, textColor=colors.HexColor("#2563eb")
    )
    normal_style = styles['Normal']
    
    elements = []
    
    # Header
    elements.append(Paragraph("AI-Assisted Chest X-Ray Clinical Report", title_style))
    elements.append(Paragraph(f"<b>Date of Analysis:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    elements.append(Spacer(1, 15))
    
    # Prediction Box (Color Coded)
    is_safe = result['severity'] == 'safe'
    bg_color = colors.HexColor("#ecfdf5") if is_safe else colors.HexColor("#fef2f2")
    border_color = colors.HexColor("#10b981") if is_safe else colors.HexColor("#ef4444")
    text_color = colors.HexColor("#065f46") if is_safe else colors.HexColor("#991b1b")
    
    pred_style = ParagraphStyle('Pred', parent=normal_style, fontSize=16, textColor=text_color, alignment=1)
    pred_data = [[Paragraph(f"<b>DIAGNOSIS: {result['predicted']}</b>", pred_style)],
                 [Paragraph(f"Calibrated Confidence: {result['calibrated_confidence']*100:.1f}%", pred_style)]]
    pred_table = Table(pred_data, colWidths=[500])
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_color),
        ('BOX', (0, 0), (-1, -1), 2, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
    ]))
    elements.append(pred_table)
    elements.append(Spacer(1, 20))
    
    # Images side-by-side
    img_orig = _base64_to_image(result['image'])
    img_heat = _base64_to_image(result['heatmap'])
    
    # Using native 224x224 size for crispness
    img_table_data = [
        [Image(img_orig, width=224, height=224), Image(img_heat, width=224, height=224)],
        [Paragraph("<b>Original (CLAHE)</b>", normal_style), Paragraph("<b>Grad-CAM Heatmap</b>", normal_style)]
    ]
    img_table = Table(img_table_data, colWidths=[250, 250])
    img_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
    elements.append(img_table)
    elements.append(Spacer(1, 20))
    
    # Clinical Details
    elements.append(Paragraph("Clinical Observations & Next Steps", heading_style))
    elements.append(Paragraph(result['explanation'], normal_style))
    elements.append(Spacer(1, 10))
    
    for step in result['advice']:
        elements.append(Paragraph(f"• {step}", normal_style))
    
    elements.append(Spacer(1, 20))
    
    # Probabilities Table
    elements.append(Paragraph("Model Probabilities", heading_style))
    prob_data = [["Class", "Raw Confidence", "Calibrated Confidence"]]
    for p in result['probabilities']:
        prob_data.append([p['name'], f"{p['raw']*100:.1f}%", f"{p['calibrated']*100:.1f}%"])
        
    prob_table = Table(prob_data, colWidths=[200, 150, 150])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f8f9fc")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#48505e")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#eef0f5")),
    ]))
    elements.append(prob_table)
    elements.append(Spacer(1, 30))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle('Disc', parent=normal_style, fontSize=9, textColor=colors.gray, alignment=1)
    elements.append(Paragraph("<i>DISCLAIMER: This is a research prototype. Not for clinical diagnosis. Please consult a radiologist for a formal interpretation.</i>", disclaimer_style))
    
    # Build PDF
    doc.build(elements)
    pdf_buffer.seek(0)
    
    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=xray_clinical_report.pdf"}
    )