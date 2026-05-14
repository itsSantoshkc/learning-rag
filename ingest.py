# ingest.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob

def load_docs(folder: str):
    docs = []
    for path in glob.glob(f"{folder}/**/*.pdf", recursive=True):
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    return docs

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # chars per chunk
        chunk_overlap=100,    # overlap prevents cutoffs
        separators=["", "", ". ", " "]
    )
    return splitter.split_documents(docs)




from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

def build_vectorstore(chunks, persist_dir="./chroma_db"):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"  # 1536-dim, cheap
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vectorstore

if __name__ == "__main__":
    docs = load_docs("./docs")
    chunks = chunk_docs(docs)
    print(f"Indexed {len(chunks)} chunks")
    build_vectorstore(chunks)