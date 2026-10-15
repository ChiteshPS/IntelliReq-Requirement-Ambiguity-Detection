import os
from PyPDF2 import PdfReader
from docx import Document
from nltk.tokenize import sent_tokenize

class DocumentParser:
    def __init__(self):
        pass
        
    def extract_requirements(self, file_path: str) -> list:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
            
        ext = file_path.split('.')[-1].lower()
        
        if ext == 'txt':
            return self._parse_txt(file_path)
        elif ext == 'pdf':
            return self._parse_pdf(file_path)
        elif ext in ['doc', 'docx']:
            return self._parse_docx(file_path)
        else:
            raise ValueError("Unsupported file format. Supported: TXT, PDF, DOCX.")
            
    def _parse_txt(self, path):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        return self._segment_text(text)
        
    def _parse_pdf(self, path):
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t + "\n"
        return self._segment_text(text)
        
    def _parse_docx(self, path):
        doc = Document(path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return self._segment_text(text)
        
    def _segment_text(self, text):
        # 1. Clean up excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 2. Tokenize by sentence
        sentences = sent_tokenize(text)
        
        # 3. Filter candidates (must look vaguely like a requirement)
        reqs = []
        for s in sentences:
            s = s.strip()
            # Ignore short artifacts, headers, page numbers
            if len(s.split()) >= 4:
                reqs.append(s)
        return reqs
