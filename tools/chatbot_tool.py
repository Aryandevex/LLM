import os
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv, find_dotenv

# =========================================================
# 1. Environment
# =========================================================
load_dotenv(find_dotenv())
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not HUGGINGFACE_API_KEY:
    raise RuntimeError("HUGGINGFACE_API_KEY missing")

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# =========================================================
# 2. LLM Setup
# =========================================================

endpoint = HuggingFaceEndpoint(
    repo_id=MODEL_ID,
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.1,
    max_new_tokens=400,
)

llm = ChatHuggingFace(llm=endpoint)

# =========================================================
# 3. TOOLS (FAMOUS, REAL-WORLD)
# =========================================================

# ---- Mock databases (simulate real systems) ----

USERS_DB = {
    "u1": {"name": "Alice", "plan": "premium"},
    "u2": {"name": "Bob", "plan": "free"},
}

INVENTORY_DB = {
    "p1": {"name": "Laptop", "stock": 12},
    "p2": {"name": "Phone", "stock": 0},
}

DOCS_DB = {
    "refund": "Refunds are processed within 7 business days.",
    "leave": "Employees get 24 days of paid leave annually.",
}

# ---- Tools ----

@tool
def run_sql_query(query: str) -> str:
    """Run read-only analytics queries."""
    return f"SQL result for query: '{query}' → 124 rows"

@tool
def search_documents(query: str) -> str:
    """Search internal documents."""
    for key, value in DOCS_DB.items():
        if key in query.lower():
            return value
    return "No relevant document found."

@tool
def get_user_profile(user_id: str) -> str:
    """Fetch user profile from CRM."""
    user = USERS_DB.get(user_id)
    if not user:
        return "User not found"
    return f"User {user['name']} is on {user['plan']} plan."

@tool
def check_inventory(product_id: str) -> str:
    """Check inventory availability."""
    product = INVENTORY_DB.get(product_id)
    if not product:
        return "Product not found"
    return f"{product['name']} stock: {product['stock']} units."

@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date info."""
    return f"Top web result for '{query}'"

# Tool registry
TOOLS = {
    "run_sql_query": run_sql_query,
    "search_documents": search_documents,
    "get_user_profile": get_user_profile,
    "check_inventory": check_inventory,
    "web_search": web_search,
}

llm_with_tools = llm.bind_tools(list(TOOLS.values()))

# =========================================================
# 4. Prompt (Chatbot Brain)
# =========================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a production chatbot for a SaaS company. "
            "Use tools to fetch real data. "
            "Never hallucinate internal information."
        ),
        ("human", "{input}")
    ]
)

chain = prompt | llm_with_tools

# =========================================================
# 5. Tool Dispatcher
# =========================================================

def execute_tool(tool_call: Dict[str, Any]) -> str:
    name = tool_call["name"]
    args = tool_call.get("args", {})

    if name not in TOOLS:
        return f"Unknown tool: {name}"

    try:
        return TOOLS[name].invoke(args)
    except Exception as e:
        return f"Tool error: {str(e)}"

# =========================================================
# 6. Chatbot Runner
# =========================================================

def chatbot(user_input: str) -> str:
    response = chain.invoke({"input": user_input})

    if response.tool_calls:
        outputs = []
        for call in response.tool_calls:
            outputs.append(execute_tool(call))
        return "\n".join(outputs)

    return response.content

# =========================================================
# 7. CLI Chatbot
# =========================================================

if __name__ == "__main__":
    print("🤖 Chatbot started (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        answer = chatbot(user_input)
        print("Bot:", answer)