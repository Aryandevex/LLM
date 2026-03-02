import os
import pandas as pd
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
    HuggingFaceEmbeddings
)

# =========================================================
# 1. ENV + CONFIG
# =========================================================
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not HUGGINGFACE_API_KEY:
    raise RuntimeError("Missing HUGGINGFACE_API_KEY")

DATA_PATH = "ecommerce_fashion.csv"
VECTOR_PATH = "faiss_index"

# =========================================================
# 2. LOAD DATA
# =========================================================
print("📦 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# =========================================================
# 3. TOOLS (Structured Operations)
# =========================================================
@tool
def average_price_by_category(category: str) -> float:
    """Returns average price of products in a given master category."""
    filtered = df[df["masterCategory"].str.contains(category, case=False)]
    return float(filtered["price"].mean())

@tool
def count_products_by_gender(gender: str) -> int:
    """Returns total number of products for a gender."""
    return int(df[df["gender"].str.lower() == gender.lower()].shape[0])

TOOLS = [average_price_by_category, count_products_by_gender]

# =========================================================
# 4. BUILD / LOAD VECTOR STORE (FAISS)
# =========================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if os.path.exists(VECTOR_PATH):
    print("📂 Loading FAISS index...")
    vectorstore = FAISS.load_local(
        VECTOR_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("🧠 Creating FAISS index...")
    documents = []
    for _, row in df.iterrows():
        text = f"""
        Product: {row['productDisplayName']}
        Category: {row['masterCategory']} > {row['subCategory']}
        Color: {row['baseColour']}
        Gender: {row['gender']}
        Price: {row['price']}
        """
        documents.append(Document(page_content=text))

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(VECTOR_PATH)
    print("✅ FAISS index saved")

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# =========================================================
# 5. LLM SETUP
# =========================================================
llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.2,
    max_new_tokens=300
)

llm = ChatHuggingFace(llm=llm_endpoint)
llm_with_tools = llm.bind_tools(TOOLS)

# =========================================================
# 6. RAG PROMPT
# =========================================================
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a smart e-commerce fashion assistant. "
     "Use the provided product context to answer. "
     "Use tools ONLY when calculations or statistics are needed."),
    ("human",
     "Question: {question}\n\nProduct Context:\n{context}")
])

# =========================================================
# 7. CHAT LOOP (Production Logic)
# =========================================================
print("\n🛒 E-Commerce Fashion Bot Ready!")
print("Type 'exit' to quit.\n")

while True:
    query = input("User: ")
    if query.lower() in {"exit", "quit"}:
        print("👋 Bye!")
        break

    # ---- RAG retrieval ----
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    # ---- LLM invoke ----
    response = llm_with_tools.invoke(
        PROMPT.format(
            question=query,
            context=context
        )
    )

    # ---- Tool execution ----
    if response.tool_calls:
        for call in response.tool_calls:
            name = call["name"]
            args = call["args"]

            print(f"\n🔧 Tool called: {name}")
            if name == "average_price_by_category":
                print("📊 Result:", average_price_by_category.invoke(args))
            elif name == "count_products_by_gender":
                print("👕 Result:", count_products_by_gender.invoke(args))
    else:
        print("\n🤖 Bot:", response.content)
