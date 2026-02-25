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
    max_new_tokens=300,
)

llm = ChatHuggingFace(llm=endpoint)

# =========================================================
# 3. Famous Tool: Payment Status
# =========================================================

# Mock database (represents Stripe / DB / internal service)
PAYMENT_DB = {
    "pay_101": {"status": "succeeded", "amount": 120.00, "currency": "USD"},
    "pay_102": {"status": "pending", "amount": 75.50, "currency": "USD"},
    "pay_103": {"status": "failed", "amount": 42.00, "currency": "USD"},
}

@tool
def check_payment_status(payment_id: str) -> str:
    """
    Check the status of a payment using its payment ID.
    Use this tool when the user asks about payment success, failure, or status.
    """
    payment = PAYMENT_DB.get(payment_id)

    if not payment:
        return f"No payment found for ID {payment_id}"

    return (
        f"Payment {payment_id} is {payment['status']}. "
        f"Amount: {payment['amount']} {payment['currency']}."
    )

TOOLS = {
    "check_payment_status": check_payment_status
}

llm_with_tools = llm.bind_tools(list(TOOLS.values()))

# =========================================================
# 4. Prompt
# =========================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a production AI assistant for a payments platform. "
            "Never guess payment data. Use tools to fetch payment status."
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
        return f"Payment service error: {str(e)}"

# =========================================================
# 6. Assistant Runner
# =========================================================

def run_assistant(user_input: str) -> str:
    response = chain.invoke({"input": user_input})

    if response.tool_calls:
        results = []
        for call in response.tool_calls:
            results.append(execute_tool(call))
        return "\n".join(results)

    return response.content

# =========================================================
# 7. Example Usage
# =========================================================

if __name__ == "__main__":
    print(run_assistant("Did payment pay_101 go through?"))
    print(run_assistant("Why did payment pay_103 fail?"))