import importlib
import importlib.util
from io import BytesIO
import os
# Import the unified cleaning function from preprocessing
from preprocessing import clean_resume_text

# Safe import of PDF libraries
PyPDF2 = importlib.import_module("PyPDF2") if importlib.util.find_spec("PyPDF2") else None
pdfplumber = importlib.import_module("pdfplumber") if importlib.util.find_spec("pdfplumber") else None

def extract_text(pdf_input):
    """
    Unified entry point. Uses the shared cleaning logic from preprocessing.py.
    """
    if isinstance(pdf_input, list):
        extracted_bundle = {}
        for index, item in enumerate(pdf_input):
            name = os.path.basename(item) if isinstance(item, str) else getattr(item, 'name', f"file_{index+1}.pdf")
            extracted_bundle[name] = extract_text(item)
        return extracted_bundle

    if isinstance(pdf_input, str):
        try:
            with open(pdf_input, 'rb') as file:
                return _extract_core(file)
        except Exception:
            return ""
    elif hasattr(pdf_input, 'read'):
        if hasattr(pdf_input, 'seek'):
            pdf_input.seek(0)
        return _extract_core(pdf_input)
    elif isinstance(pdf_input, bytes):
        return _extract_core(BytesIO(pdf_input))
    return ""

def _extract_core(pdf_handle):
    text = ""
    if pdfplumber:
        try:
            with pdfplumber.open(pdf_handle) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text: text += page_text + "\n"
        except Exception: pass
            
    if not text and PyPDF2:
        try:
            pdf_handle.seek(0)
            reader = PyPDF2.PdfReader(pdf_handle)
            text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        except Exception: pass
        
    # Using the shared clean_resume_text from preprocessing.py
    return clean_resume_text(text)

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" SYNCED PDF EXTRACTION TEST ")
    print("="*60)
    pdfs = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    if pdfs:
        results = extract_text(pdfs)
        for name, content in results.items():
            print(f"[ {name} ] -> {len(content)} chars (Cleaned via preprocessing.py)")
    else:
        print("No PDFs found.")
    print("\n" + "="*60)