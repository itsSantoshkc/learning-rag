# chain.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use ONLY the context below
to answer. If the answer isn't in the context, say so.

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def build_chain(retriever):
    # Replaced ChatOpenAI with ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model= "gemini-3-flash-preview",
        temperature=0,
        # 'flash' is fast and great for RAG, use 'pro' for complex reasoning
    )
    
    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | PROMPT 
        | llm 
        | StrOutputParser()
    )