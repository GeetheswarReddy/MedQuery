from pypdf import PdfReader
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_and_chunk_pdf(pdf_path: str,chunk_size: int =1000,chunk_overlap: int = 50) -> list[Document]:
    # Load PDF
    path=Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    reader = PdfReader(str(pdf_path))

    raw_pages = []
    for page_num,page in enumerate(reader.pages):
        text=page.extract_text()
        if text:
            text=text.strip()
            raw_pages.append(Document(
                page_content=text,
                metadata={"source": path.name,"page": page_num+1}
            )
        )
    # Chunking
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n","\n","."," ",""]
    )
    chunks=splitter.split_documents(raw_pages)

    print(f"PDF: {path.name}")
    print(f"Pages loaded: {len(raw_pages)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Avg chunk length: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
    
    return chunks

