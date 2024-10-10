import os
from docx import Document
import PyPDF2

def extract_text_from_word(docx_path):
    doc = Document(docx_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return "\n".join(full_text)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        full_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    return "\n".join(full_text)

def save_extracted_text(file_name, text, output_dir='data/extracted_text/'):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Extracted text saved to {txt_path}")

def extract_all_documents(documents_dir='data/documents/'):
    for file in os.listdir(documents_dir):
        file_path = os.path.join(documents_dir, file)
        if file.endswith('.docx'):
            text = extract_text_from_word(file_path)
            save_extracted_text(file, text)
        elif file.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
            save_extracted_text(file, text)
        else:
            print(f"Unsupported file format: {file}")

if __name__ == "__main__":
    extract_all_documents()