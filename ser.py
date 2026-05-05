# react_agent_serpapi.py

import os
from dotenv import load_dotenv, find_dotenv

# LangChain imports
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# SerpAPI
from serpapi import GoogleSearch

# -----------------------------
# 🔐 Load API Keys
# -----------------------------
load_dotenv(find_dotenv())

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# -----------------------------
# 🤖 LLM Setup
# -----------------------------
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        temperature=0.1,
        max_new_tokens=256
    )
)

# -----------------------------
# 🛠️ Calculator Tool
# -----------------------------
def calculator_tool(query: str):
    try:
        return str(eval(query))
    except Exception as e:
        return f"Error: {str(e)}"

calc_tool = Tool(
    name="Calculator",
    func=calculator_tool,
    description="""
Use this tool for math calculations.

STRICT RULES:
- Always use this tool for math
- Do NOT calculate yourself
- Format:
  Action: Calculator
  Action Input: <expression>
"""
)

# -----------------------------
# 🌐 SerpAPI Search Tool
# -----------------------------
def search_tool(query: str):
    try:
        params = {
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "engine": "google",
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        if "organic_results" in results:
            snippets = []
            for r in results["organic_results"][:3]:
                snippets.append(r.get("snippet", ""))
            return "\n".join(snippets)

        return "No results found"

    except Exception as e:
        return f"Error: {str(e)}"

search_tool_obj = Tool(
    name="Search",
    func=search_tool,
    description="""
Use this tool to search the internet.

Use it for:
- latest news
- current events
- real-world facts

STRICT RULES:
- Always use this for unknown or current info
- Do NOT answer from memory

Format:
Action: Search
Action Input: <query>
"""
)

# -----------------------------
# 🔗 Tools List
# -----------------------------
tools = [calc_tool, search_tool_obj]

# -----------------------------
# 🧠 Prompt (ReAct)
# -----------------------------
prompt = hub.pull("hwchase17/react")

# Make it stricter for Qwen
prompt.template += """

IMPORTANT RULES:
- Follow format strictly
- Do NOT give Final Answer before using tools
- Use Calculator for math
- Use Search for real-time or unknown info
- Always take one step at a time
"""

# -----------------------------
# 🤖 Create Agent
# -----------------------------
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# -----------------------------
# 🚀 Agent Executor
# -----------------------------
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# -----------------------------
# ▶️ Run
# -----------------------------
if __name__ == "__main__":

    # Try different queries
    query = "What is the latest news about AI in 2026?"
    # query = "What is 25 * 4?"

    response = agent_executor.invoke({
        "input": query
    })

    print("\nFinal Output:")
    print(response["output"])
