import os
from typing import Dict, Any
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv, find_dotenv

# 1. Environment
load_dotenv(find_dotenv())
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.1
))

# 2. Level 3 Tools: CRM
CUSTOMERS = {
    "c1": {"name": "Aryan", "email": "aryan@example.com", "tier": "Gold"},
    "c2": {"name": "John", "email": "john@example.com", "tier": "Silver"}
}

@tool
def get_customer_details(customer_id: str) -> str:
    """Lookup customer profile in CRM by ID (e.g., c1, c2)."""
    customer = CUSTOMERS.get(customer_id)
    if not customer: return "Customer not found."
    return f"Name: {customer['name']}, Tier: {customer['tier']}"

@tool
def check_loyalty_points(customer_id: str) -> str:
    """Checks the loyalty points for a customer."""
    points = {"c1": 4500, "c2": 1200}
    return f"Customer {customer_id} has {points.get(customer_id, 0)} points."

# 3. Tool Registry & Binding
TOOL_MAP = {
    "get_customer_details": get_customer_details,
    "check_loyalty_points": check_loyalty_points
}

llm_with_tools = llm.bind_tools(list(TOOL_MAP.values()))

# 4. Generic Execution Logic
def execute_tools(tool_calls):
    results = []
    for call in tool_calls:
        name = call["name"]
        args = call["args"]
        print(f"--- Executing {name} with {args} ---")
        if name in TOOL_MAP:
            result = TOOL_MAP[name].invoke(args)
            results.append(result)
    return results

# 5. Assistant Chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a CRM Specialist. Always check customer details before answering."),
    ("human", "{input}")
])

chain = prompt | llm_with_tools

def crm_bot(query: str):
    print(f"User: {query}")
    response = chain.invoke({"input": query})
    
    if response.tool_calls:
        tool_results = execute_tools(response.tool_calls)
        print(f"System Responses: {tool_results}")
    else:
        print(f"Bot: {response.content}")

if __name__ == "__main__":
    print("--- Level 3: CRM Manager ---")
    crm_bot("Tell me about customer c1.")
    crm_bot("How many points does c2 have?")
