from retriever import load_retriever
from chain import build_chain
from dotenv import load_dotenv

load_dotenv()

def main():
    retriever = load_retriever()
    chain = build_chain(retriever)

    print("RAG app ready. Type 'exit' to quit.")
    while True:
        q = input("You: ").strip()
        if q.lower() == "exit": break
        print("Bot: ", end="", flush=True)
        for chunk in chain.stream(q):
            print(chunk, end="", flush=True)
        print("")

if __name__ == "__main__":
    main()


    