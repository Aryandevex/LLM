import os
from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# ---------------------------
# 1. ENV
# ---------------------------
load_dotenv(find_dotenv())
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# ---------------------------
# 2. Load PDF
# ---------------------------
pdf_path = "sample.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# ---------------------------
# 3. Split into chunks
# ---------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# ---------------------------
# 4. Embeddings (FREE local)
# ---------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------
# 5. Vector DB
# ---------------------------
vectorstore = FAISS.from_documents(chunks, embeddings)

# ---------------------------
# 6. Retriever
# ---------------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------------------
# 7. LLM (HF Free Endpoint)
# ---------------------------
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",  # you can change
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        temperature=0.2,
        max_new_tokens=300
    )
)

# ---------------------------
# 8. Prompt
# ---------------------------
prompt = ChatPromptTemplate.from_template("""
Answer ONLY from the given context.
If answer not found, say "Not in document".

Context:
{context}

Question:
{question}
""")

# ---------------------------
# 9. RAG Chain
# ---------------------------
def rag_chain(question):
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content


# ---------------------------
# 10. Run
# ---------------------------
if __name__ == "__main__":
    print("RAG Ready 🔥\n")

    while True:
        q = input("Ask: ")

        if q.lower() in ["exit", "quit"]:
            break

        ans = rag_chain(q)
        print("\nAnswer:\n", ans)
        print("\n" + "-"*50)
