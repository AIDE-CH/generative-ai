
#%%
"""
python -m pip install pymupdf4llm
"""

import pymupdf4llm
import os

def extract_text_from_pdf(pdf_file):
    pdf_path = "pdfs/" + pdf_file
    output_file = "markdown/" + pdf_file.replace(".pdf", ".md")
    with open(output_file, "w", encoding="utf-8") as out_file:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        md_clean = clean_pdf_text(md_text)
        out_file.write(md_clean)

def clean_pdf_text(text):
    #text = text.encode("ascii", "ignore").decode()  # Remove non-ASCII
    text = text.replace("�", "")  # Remove unknown symbols
    text = text.replace("", " ")  # Remove unknown symbols
    #text = re.sub(r"[^\x20-\x7E]", "", text)  # Remove non-printable characters
    #text = "".join(c for c in text if unicodedata.category(c)[0] != "C")  # Normalize Unicode
    return text

#%%

pdf_files = [f for f in os.listdir("pdfs/") if f.endswith(".pdf")]

for file in pdf_files:  
    extract_text_from_pdf(file)
