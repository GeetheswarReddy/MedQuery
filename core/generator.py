from chunker import load_and_chunk_pdf
from embedder import index_chunks, load_index
from pathlib import Path

path=Path(__file__).parent.parent / "data" / "sample_docs" /"sample.pdf"
chunks = load_and_chunk_pdf(path)
index=index_chunks(chunks)
print(f"Index saved at: {Path('data/faiss_index').resolve()}")