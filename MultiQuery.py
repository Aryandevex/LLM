# Multi-Query Retrieval Practical Example using LangChain

# Install:
# pip install langchain langchain-community langchain-openai faiss-cpu sentence-transformers

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

# -----------------------------------
# Sample Documents
# -----------------------------------

docs = [
    "RAG systems need proper chunking strategy.",
    "Hybrid search improves retrieval accuracy.",
    "Vector databases require embedding optimization.",
    "BM25 helps keyword-based retrieval.",
    "Smaller chunks improve LLM context quality.",
    "QLoRA reduces GPU memory usage for fine-tuning."
]

# Save docs into file
with open("data.txt", "w") as f:
    for doc in docs:
        f.write(doc + "\n")

# -----------------------------------
# Load Documents
# -----------------------------------

loader = TextLoader("data.txt")
documents = loader.load()

# -----------------------------------
# Split Documents
# -----------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

split_docs = text_splitter.split_documents(documents)

# -----------------------------------
# Create Embeddings
# -----------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------------
# Store in Vector DB
# -----------------------------------

vector_db = FAISS.from_documents(split_docs, embeddings)

# -----------------------------------
# Create Retriever
# -----------------------------------

retriever = vector_db.as_retriever()

# -----------------------------------
# LLM
# -----------------------------------

llm = ChatOpenAI(
    temperature=0,
    api_key="YOUR_OPENAI_API_KEY"
)

# -----------------------------------
# Multi Query Retriever
# -----------------------------------

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

# -----------------------------------
# Query
# -----------------------------------

query = "How can I improve RAG performance?"

results = multi_query_retriever.invoke(query)

# -----------------------------------
# Output
# -----------------------------------

print("\nRetrieved Documents:\n")

for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}")
