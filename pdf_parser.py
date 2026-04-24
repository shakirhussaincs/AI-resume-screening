import importlib
import importlib.util
from io import BytesIO
import re
import os

# Safe import of PDF libraries
PyPDF2 = importlib.import_module("PyPDF2") if importlib.util.find_spec("PyPDF2") else None
pdfplumber = importlib.import_module("pdfplumber") if importlib.util.find_spec("pdfplumber") else None

def extract_text(pdf_input):
    """
    The UNIFIED entry point.
    - If pdf_input is a LIST: returns a dictionary {filename: text}
    - If pdf_input is a SINGLE file (path, bytes, object): returns a string (text)
    """
    # Detect if it's a bundle (list)
    if isinstance(pdf_input, list):
        extracted_bundle = {}
        for index, item in enumerate(pdf_input):
            if isinstance(item, str):
                name = os.path.basename(item)
            else:
                name = getattr(item, 'name', f"file_{index + 1}.pdf")
            
            # Recursively call extract_text for each item in the list
            extracted_bundle[name] = extract_text(item)
        return extracted_bundle

    # Otherwise, process as a single file
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
    """Internal helper to choose the best library for extraction."""
    if pdfplumber:
        try:
            full_text = ""
            with pdfplumber.open(pdf_handle) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
            return _clean_text(full_text)
        except Exception:
            pass # Fallback to PyPDF2
            
    if PyPDF2:
        try:
            pdf_handle.seek(0)
            reader = PyPDF2.PdfReader(pdf_handle)
            return _clean_text("\n".join([p.extract_text() for p in reader.pages if p.extract_text()]))
        except Exception:
            return ""
    return ""

def _clean_text(text):
    if not text: return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.,;:!?\-\(\)\[\]\{\}/@#$%&*+=]', '', text)
    return text.strip()

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" UNIFIED PDF EXTRACTION TEST ")
    print("="*60)
    
    # Setup test data
    all_pdfs = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    
    if all_pdfs:
        # --- TEST 1: MULTIPLE FILES (BUNDLE) ---
        print("\n>>> TESTING MULTIPLE FILES (BUNDLE)")
        print(f"Input: List of {len(all_pdfs)} files")
        bundle_results = extract_text(all_pdfs) # Same function!
        
        for name, content in bundle_results.items():
            safe_preview = "".join(c for c in content[:70] if c.isprintable())
            print(f"  [ {name} ] -> {len(content)} characters | Preview: {safe_preview}...")
        
        # --- TEST 2: SINGLE FILE ---
        print("\n" + "-"*60)
        print(">>> TESTING SINGLE FILE")
        single_file = all_pdfs[0]
        print(f"Input: Single string path ('{single_file}')")
        single_result = extract_text(single_file) # Same function!
        
        print(f"  Result: Successfully extracted {len(single_result)} characters.")
        safe_preview = "".join(c for c in single_result[:70] if c.isprintable())
        print(f"  Preview: {safe_preview}...")
        
    else:
        print("No PDF files found for testing.")
    
    print("\n" + "="*60)