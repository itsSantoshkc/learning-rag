from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import glob
import os

# Load environment variables (GOOGLE_API_KEY)
load_dotenv()

def load_docs(folder: str):
    """Loads all PDF documents from the specified folder."""
    docs = []
    # Search for PDFs in the folder and subdirectories
    pdf_files = glob.glob(f"{folder}/**/*.pdf", recursive=True)
    
    if not pdf_files:
        print(f"⚠️ No PDFs found in {folder}")
        return []

    for path in pdf_files:
        try:
            print(f"📄 Loading: {path}")
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"❌ Failed to load {path}: {e}")
    return docs

def chunk_docs(docs):
    """Splits documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(docs)

def build_vectorstore(chunks, persist_dir="./chroma_db"):
    """Creates a Chroma vectorstore using Google GenAI embeddings."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("Missing GOOGLE_API_KEY. Please check your .env file.")

    # FIX: Remove 'models/' prefix. 
    # Just use 'text-embedding-004'.
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        task_type="retrieval_document"
    )

    print(f"📦 Building vector store at {persist_dir}...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    return vectorstore

if __name__ == "__main__":
    # 1. Load
    raw_docs = load_docs("./docs")
    
    if raw_docs:
        # 2. Chunk
        chunks = chunk_docs(raw_docs)
        print(f"✅ Created {len(chunks)} chunks.")
        
        # 3. Embed and Store
        build_vectorstore(chunks)
        print("🚀 Vector store successfully built and saved!")