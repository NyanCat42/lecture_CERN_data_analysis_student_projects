import easyocr
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
import numpy as np
import io


def extract_text_from_pdf(pdf_path, output_path):
    """
    Extract text from PDF using EasyOCR
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path to save the extracted text
    """
    print(f"Processing: {pdf_path}")
    
    # Initialize EasyOCR reader (first time will download the model)
    print("Initializing OCR reader...")
    reader = easyocr.Reader(['en', 'lt'])  # Add more languages if needed: ['en', 'lt']
    
    # Open the PDF
    doc = fitz.open(pdf_path)
    print(f"Found {len(doc)} pages")
    
    # Extract text from each page
    all_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Convert page to image
        pix = page.get_pixmap(dpi=300)  # Higher DPI for better OCR
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        # Convert PIL Image to numpy array for EasyOCR
        img_array = np.array(img)
        
        print(f"Processing page {page_num + 1}/{len(doc)} with OCR...")
        
        # Perform OCR
        results = reader.readtext(img_array, detail=0)  # detail=0 returns only text
        text = "\n".join(results)
        
        all_text.append(f"--- Page {page_num + 1} ---\n{text}\n")
    
    doc.close()
    
    # Save to file
    full_text = "\n".join(all_text)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"Text extracted and saved to: {output_path}")
    return full_text


if __name__ == "__main__":
    # Set up paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"
    output_dir = project_dir / "output"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Input and output files
    pdf_file = data_dir / "lit-hb-price-list-11.2025.pdf"
    output_file = output_dir / "menu_text.txt"
    
    # Extract text
    text = extract_text_from_pdf(pdf_file, output_file)
    
    print("\n--- First 500 characters of extracted text ---")
    print(text[:500])
