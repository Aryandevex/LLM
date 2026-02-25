import os
from dotenv import load_dotenv, find_dotenv

import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# -----------------------------
# 1. Load API Key
# -----------------------------
load_dotenv(find_dotenv())
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")

# -----------------------------
# 2. LLM Setup (No Deprecation)
# -----------------------------
endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.1,
    max_new_tokens=400,
)

llm = ChatHuggingFace(llm=endpoint)

# -----------------------------
# 3. Define Tool
# -----------------------------
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b

# -----------------------------
# 4. Bind Tool to LLM
# -----------------------------
llm_with_tools = llm.bind_tools([add_numbers])

# -----------------------------
# 5. Prompt Template
# -----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that uses tools when needed."),
        ("human", "{input}")
    ]
)

# -----------------------------
# 6. Build Chain
# -----------------------------
chain = prompt | llm_with_tools

# -----------------------------
# 7. Invoke Model
# -----------------------------
response = chain.invoke(
    {
        "input": "What is 25 plus 17?"
    }
)

# -----------------------------
# 8. Handle Tool Calls
# -----------------------------
if response.tool_calls:
    for tool_call in response.tool_calls:
        if tool_call["name"] == "add_numbers":
            args = tool_call["args"]
            result = add_numbers.invoke(args)
            print("Tool result:", result)
else:
    print("Model response:", response.content)