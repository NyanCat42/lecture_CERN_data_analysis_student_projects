from pathlib import Path
import requests


def download_menu_pdf(pdf_path):
    """
    Download the menu PDF from Hesburger website if not present.
    
    Args:
        pdf_path: Path where the PDF should be saved
    """
    if pdf_path.exists():
        print(f"PDF already exists at {pdf_path}")
        return
    
    print(f"PDF not found. Downloading from Hesburger website...")
    
    url = "https://www.hesburger.lt/mellow/output/getfile.php?id=2770"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Create data directory if it doesn't exist
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the PDF
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Successfully downloaded PDF to {pdf_path}")
        print(f"File size: {len(response.content) / 1024:.2f} KB")
        
    except requests.RequestException as e:
        print(f"Error downloading PDF: {e}")
        raise


if __name__ == "__main__":
    # This block only runs when the script is executed directly
    from pathlib import Path
    
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"
    
    pdf_file = data_dir / "lit-hb-price-list-11.2025.pdf"
    download_menu_pdf(pdf_file)
