from io import BytesIO
import PyPDF2
import docx
from PIL import Image

def extract_text_from_pdf(file_bytes):
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def extract_text_from_docx(file_bytes):
    doc = docx.Document(BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])

def load_image(file_bytes):
    return Image.open(BytesIO(file_bytes)).convert("RGB")
