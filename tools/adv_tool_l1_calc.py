import os
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv, find_dotenv

# 1. Environment Support
load_dotenv(find_dotenv())
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not HUGGINGFACE_API_KEY:
    raise RuntimeError("HUGGINGFACE_API_KEY missing from .env")

# 2. Setup LLM
llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.1,
    max_new_tokens=100,
)
llm = ChatHuggingFace(llm=llm_endpoint)

# 3. Level 1 Tool: Basic Calculator
@tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers and returns the result."""
    return a * b

# 4. Bind Tool
llm_with_tools = llm.bind_tools([multiply])

# 5. Simple Chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a math assistant. Use tools for calculations."),
    ("human", "{input}")
])

chain = prompt | llm_with_tools

# 6. Execution Loop (Simple)
if __name__ == "__main__":
    query = "What is 123 multiplied by 456?"
    print(f"User: {query}")
    
    response = chain.invoke({"input": query})
    
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"--- Calling tool: {tool_call['name']} ---")
            result = multiply.invoke(tool_call["args"])
            print(f"Result: {result}")
    else:
        print(f"Bot: {response.content}")
