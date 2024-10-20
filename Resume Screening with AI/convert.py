import fitz
from io import BytesIO
import streamlit as st

def ExtractPDFText(pdf):
    content = ""
    pdf_bytes = pdf.read()

    try:
        pdf_document = fitz.open("dummy.pdf", pdf_bytes)
        
        # Iterate through pages and extract text
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            text = page.get_text()
            content += text
        
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        
    finally:
        if "pdf_document" in locals():
            pdf_document.close()

    return content


