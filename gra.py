# program_1_simple_graph.py
"""
A bare-minimum LangGraph that just passes data through 2 nodes.
Think of it as a pipeline: Input -> Process -> Transform -> Output
"""

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# STEP 1: Define what data flows through your graph
class State(TypedDict):
    message: str
    counter: int

# STEP 2: Define nodes (each node receives state, returns updates)
def uppercase_node(state: State):
    print(f"📥 Uppercase node received: {state['message']}")
    return {"message": state["message"].upper()}

def exclaim_node(state: State):
    print(f"📥 Exclaim node received: {state['message']}")
    return {"message": state["message"] + "!!!"}

def count_node(state: State):
    print(f"📥 Count node received: {state['message']}")
    return {"counter": state["counter"] + len(state["message"])}

# STEP 3: Build the graph
builder = StateGraph(State)

# Add nodes
builder.add_node("uppercase", uppercase_node)
builder.add_node("exclaim", exclaim_node)
builder.add_node("count", count_node)

# Add edges (defines the flow)
builder.add_edge(START, "uppercase")  # Start -> uppercase
builder.add_edge("uppercase", "exclaim")  # uppercase -> exclaim
builder.add_edge("exclaim", "count")  # exclaim -> count
builder.add_edge("count", END)  # count -> End

# STEP 4: Compile (makes it executable)
graph = builder.compile()

# STEP 5: Run it!
result = graph.invoke({
    "message": "hello world",
    "counter": 0
})

print("\n" + "="*50)
print(f"✅ Final Result: {result}")
print("="*50)

"""
EXPECTED OUTPUT:
📥 Uppercase node received: hello world
📥 Exclaim node received: HELLO WORLD
📥 Count node received: HELLO WORLD!!!
✅ Final Result: {'message': 'HELLO WORLD!!!', 'counter': 14}
"""
