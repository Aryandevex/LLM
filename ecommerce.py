import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv

# LangChain Imports
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. SETUP & CONFIGURATION
load_dotenv(find_dotenv())
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY not found in environment variables.")

DATASET_PATH = r"C:\Users\Lenovo\Downloads\archive (3)\FashionDataset.csv"

def run_rag_bot():
    print("--- Starting E-commerce RAG Bot Pipeline ---")

    # 2. LOAD & PREPROCESS DATA
    print("\n[Step 1/5] Loading and cleaning dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    # We'll take a sample of 1000 products for a quick demonstration.
    # You can increase this to include the full 50k+ dataset if your machine allows.
    df_sample = df.sample(n=min(1000, len(df)), random_state=42).fillna("N/A")

    # Convert each row into a single descriptive string (Document)
    documents = []
    for _, row in df_sample.iterrows():
        content = (
            f"Brand: {row['BrandName']}. "
            f"Category: {row['Category']}. "
            f"Product: {row['Deatils']}. "
            f"Available Sizes: {row['Sizes']}. "
            f"MRP: {row['MRP']}. "
            f"Selling Price: {row['SellPrice']}. "
            f"Discount: {row['Discount']}."
        )
        # We store metadata so we can display it if needed
        doc = Document(page_content=content, metadata={"brand": row['BrandName'], "category": row['Category']})
        documents.append(doc)

    # 3. EMBEDDINGS & VECTOR DATABASE
    print("\n[Step 2/5] Creating local Vector Database (Indexing)...")
    # Using a small, fast local embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Build FAISS index (this stays on your local machine)
    vector_db = FAISS.from_documents(documents, embeddings)
    print("Vector Database initialized successfully.")

    # 4. LLM SETUP
    print("\n[Step 3/5] Initializing Zephyr Generator (Hugging Face API)...")
    endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        temperature=0.2,
        max_new_tokens=512,
    )
    llm = ChatHuggingFace(llm=endpoint)

    # 5. RAG CHAIN SETUP
    print("\n[Step 4/5] Setting up Retrieval-QA Chain...")
    
    # Custom prompt to ensure the bot answers nicely
    template = """You are a helpful e-commerce shopping assistant. 
    Use the following pieces of product context to answer the user's question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    
    Question: {question}
    
    Assistant's Response:"""
    
    rag_prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Create the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": rag_prompt}
    )

    # 6. INTERACTIVE CHAT
    print("\n[Step 5/5] RAG Pipeline Ready!")
    print("\nAsk me anything about the products (e.g., 'What indigo dresses do you have?')")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            break
        
        print("Searching and generating answer...")
        try:
            result = qa_chain.invoke(query)
            print(f"\nBot: {result['result']}\n")
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    run_rag_bot()
