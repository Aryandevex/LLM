# modern_router.py

import os
from dotenv import load_dotenv, find_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch

# -----------------------------
# 🔐 Load ENV
# -----------------------------
load_dotenv(find_dotenv())
HF_KEY = os.getenv("HUGGINGFACE_API_KEY")

# -----------------------------
# 🤖 LLM
# -----------------------------
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        huggingfacehub_api_token=HF_KEY,
        temperature=0.1,
        max_new_tokens=256
    )
)

parser = StrOutputParser()

# -----------------------------
# 🧠 Chains
# -----------------------------

# Math Chain
math_chain = (
    PromptTemplate.from_template("Solve this math problem: {input}")
    | llm
    | parser
)

# Search Chain (placeholder — plug SerpAPI later)
search_chain = (
    PromptTemplate.from_template(
        "Search the internet and answer accurately: {input}"
    )
    | llm
    | parser
)

# General Chain
general_chain = (
    PromptTemplate.from_template(
        "Answer clearly and concisely: {input}"
    )
    | llm
    | parser
)

# -----------------------------
# 🔀 Router Logic (Classifier)
# -----------------------------
def classify(x):
    text = x["input"].lower()

    if any(op in text for op in ["+", "-", "*", "/"]):
        return "math"
    elif any(word in text for word in ["latest", "news", "today", "current"]):
        return "search"
    else:
        return "general"

# -----------------------------
# 🚀 Router (Modern Way)
# -----------------------------
router = RunnableBranch(
    # condition, chain
    (lambda x: classify(x) == "math", math_chain),
    (lambda x: classify(x) == "search", search_chain),
    general_chain  # default
)

# -----------------------------
# ▶️ Run
# -----------------------------
if __name__ == "__main__":
    while True:
        query = input("\nAsk: ")

        result = router.invoke({"input": query})
        print("\nAnswer:", result)
