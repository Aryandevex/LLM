# react_agent.py

import os
from dotenv import load_dotenv, find_dotenv

# LangChain imports
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# -----------------------------
# 🔐 Load API Key
# -----------------------------
load_dotenv(find_dotenv())
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# -----------------------------
# 🤖 LLM Setup (Your Model)
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
# 🛠️ Tool (Calculator)
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
  Action Input: <math expression>
"""
)

tools = [calc_tool]

# -----------------------------
# 🧠 Prompt (ReAct from Hub)
# -----------------------------
prompt = hub.pull("hwchase17/react")

# 🔧 Make prompt stricter (IMPORTANT for Qwen)
prompt.template += """

IMPORTANT RULES:
- Follow format strictly
- Do NOT give Final Answer before using tools
- Always take one step at a time
"""

# -----------------------------
# 🤖 Create Agent (NEW API)
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
    handle_parsing_errors=True   # ✅ Fix your error
)

# -----------------------------
# ▶️ Run
# -----------------------------
if __name__ == "__main__":
    query = "What is 25 * 4?"

    response = agent_executor.invoke({
        "input": query
    })

    print("\nFinal Output:")
    print(response["output"])
