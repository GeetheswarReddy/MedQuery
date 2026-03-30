from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path

def embedding_model():
    embedding=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embedding

def index_chunks(chunks: list[Document],index_path: str="data/faiss_index") -> FAISS:
    embedding=embedding_model()
    index=FAISS.from_documents(chunks,embedding)
    Path(index_path).mkdir(parents=True, exist_ok=True)
    index.save_local(index_path)
    return index

def load_index(index_path="data/faiss_index")->FAISS:
    embedding=embedding_model()
    index=FAISS.load_local(index_path,embedding,allow_dangerous_deserialization=True)
    return index
