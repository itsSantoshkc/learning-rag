# retriever.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def load_retriever(persist_dir="./chroma_db", k=4):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    # MMR: max marginal relevance — diverse results
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20}
    )

# test it
if __name__ == "__main__":
    retriever = load_retriever()
    docs = retriever.invoke("What is the refund policy?")
    for d in docs:
        print(d.page_content[:200], "---")