import os
import json
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv, find_dotenv

# -----------------------------
# 1. Load API Key
# -----------------------------
load_dotenv(find_dotenv())
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY not found")

# -----------------------------
# 2. LLM Setup
# -----------------------------
endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.1,
    max_new_tokens=400,
)

llm = ChatHuggingFace(llm=endpoint)

# -----------------------------
# 3. Real-Time Tool
# -----------------------------
@tool
def get_current_time() -> str:
    """Get the current system time in HH:MM:SS format."""
    return datetime.now().strftime("%H:%M:%S")

# -----------------------------
# 4. Bind Tool
# -----------------------------
llm_with_tools = llm.bind_tools([get_current_time])

# -----------------------------
# 5. Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Use tools when real-time data is needed."),
        ("human", "{input}")
    ]
)

# -----------------------------
# 6. Chain
# -----------------------------
chain = prompt | llm_with_tools

# -----------------------------
# 7. Invoke
# -----------------------------
response = chain.invoke(
    {
        "input": "What is the current time right now?"
    }
)

# -----------------------------
# 8. Tool Execution
# -----------------------------
if response.tool_calls:
    for tool_call in response.tool_calls:
        if tool_call["name"] == "get_current_time":
            result = get_current_time.invoke({})
            print("Current Time:", result)
else:
    print(response.content)