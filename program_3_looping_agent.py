# program_3_looping_agent.py
"""
An agent that can reason and take actions in a loop until it has an answer.
This mimics the ReAct (Reason + Act) pattern.
"""

from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI  # pip install langchain-openai
from langchain.tools import tool
import json

# NOTE: You need an OpenAI API key. Set it as environment variable:
# export OPENAI_API_KEY="your-key-here"
# Or use a free model like Ollama (change the model line)

# === STEP 1: Define Tools ===
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together. Use this for any addition operation."""
    print(f"🔧 [TOOL] Adding {a} + {b}")
    return a + b

@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers. Use this for any multiplication operation."""
    print(f"🔧 [TOOL] Multiplying {a} × {b}")
    return a * b

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city. Use this for weather questions."""
    # Mock weather data
    weather_data = {
        "new york": "72°F and sunny",
        "london": "58°F and cloudy",
        "tokyo": "65°F and rainy",
    }
    result = weather_data.get(city.lower(), f"Unknown city: {city}")
    print(f"🔧 [TOOL] Weather for {city}: {result}")
    return result

tools = [add_numbers, multiply_numbers, get_weather]

# === STEP 2: Define State ===
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Special type for chat history
    iteration_count: int

# === STEP 3: Create LLM with Tools ===
# For OpenAI:
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# For free alternative (Ollama): llm = ChatOllama(model="llama2")

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# === STEP 4: Define Nodes ===
def call_llm(state: AgentState):
    """Call the LLM to decide what to do"""
    print(f"\n🤔 [ITERATION {state['iteration_count']}] LLM Thinking...")
    response = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [response],
        "iteration_count": state["iteration_count"] + 1
    }

def call_tool(state: AgentState):
    """Execute the tool requested by LLM"""
    last_message = state["messages"][-1]
    
    # Parse tool calls
    tool_calls = last_message.tool_calls
    
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Find and execute the right tool
        for tool in tools:
            if tool.name == tool_name:
                result = tool.invoke(tool_args)
                results.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": tool_name,
                    "content": str(result)
                })
                break
    
    return {"messages": results}

# === STEP 5: Router (Decide if we need more tool calls) ===
def should_continue(state: AgentState):
    """Determine if we should continue calling tools or finish"""
    last_message = state["messages"][-1]
    
    # If LLM wants to call tools, continue
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    else:
        return "end"

# === STEP 6: Build Graph ===
builder = StateGraph(AgentState)

builder.add_node("agent", call_llm)
builder.add_node("tools", call_tool)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {
    "continue": "tools",
    "end": END
})
builder.add_edge("tools", "agent")  # Loop back to agent!

graph = builder.compile()

# === STEP 7: Run the Agent ===
print("\n" + "="*60)
print("🤖 REACT AGENT RUNNING")
print("="*60)

# Test queries
queries = [
    "What is 25 + 17?",
    "Calculate 8 * 6 for me",
    "What's the weather in London?"
]

for query in queries:
    print(f"\n📝 User: {query}")
    result = graph.invoke({
        "messages": [("user", query)],
        "iteration_count": 0
    })
    
    print(f"\n✨ Final Answer: {result['messages'][-1].content}")
    print(f"📊 Total iterations: {result['iteration_count']}")
    print("-"*40)

"""
EXPECTED BEHAVIOR:
1. User asks "25 + 17"
2. LLM decides to call add_numbers(25, 17)
3. Tool executes, returns 42
4. LLM sees result, formulates answer
5. Returns "25 + 17 equals 42"
"""
