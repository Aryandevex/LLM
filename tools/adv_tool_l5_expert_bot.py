import os
import json
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv, find_dotenv

# 1. Setup
load_dotenv(find_dotenv())
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.1
))

# 2. Level 5 Tools: System Experts
@tool
def code_executor(command: str) -> str:
    """Simulates executing a system command or code snippet."""
    return f"Execution of '{command}' succeeded. Output: Hello World."

@tool
def file_search(filename: str) -> str:
    """Simulates searching for a file in the project workspace."""
    if "config" in filename:
        return f"File '{filename}' found. It contains API_URL=http://localhost:8000"
    return "File not found."

TOOLS = {"code_executor": code_executor, "file_search": file_search}
llm_with_tools = llm.bind_tools(list(TOOLS.values()))

# 3. Expert Agent Loop (The Logic)
def run_expert_agent(user_query: str):
    print(f"--- Expert Dev Assistant ---")
    messages = [HumanMessage(content=user_query)]
    
    # Simple loop to simulate agentic 'thinking' and 'acting'
    for i in range(3): # Limit to 3 rounds
        print(f"Round {i+1}...")
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            print(f"Assistant: {response.content}")
            break
            
        for tool_call in response.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            print(f"  > Agent wants to use: {name}({args})")
            
            # Execute tool
            if name in TOOLS:
                result = TOOLS[name].invoke(args)
                print(f"  > Tool Result: {result}")
                # We add the result back to history so the LLM can use it
                messages.append(ToolMessage(tool_call_id=tool_call["id"], content=str(result)))
            else:
                messages.append(ToolMessage(tool_call_id=tool_call["id"], content="Error: Tool unknown"))

if __name__ == "__main__":
    print("--- Level 5: Expert Bot (Multi-turn Tool Interaction) ---")
    run_expert_agent("Find the config file and then tell me how to execute it.")
