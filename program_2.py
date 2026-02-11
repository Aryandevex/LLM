from typing import TypedDict
class state(TypedDict):
    message:str
    result:str
def greeting_node(state:state):
    print("greeting node is running")
    return {
        "result": "Hello! Nice to meet you 😊"
    }
def normal_node(state:state):
    print("normal node is running")
    return {
        "result": f"You said: {state['message']}"
    }
from langgraph.graph import StateGraph,END,START
graph_builder=StateGraph(state)
graph_builder.add_node("greeting_node",greeting_node)
graph_builder.add_node("normal_node",normal_node)
graph_builder.add_edge(START,"greeting_node")
graph_builder.add_edge("greeting_node",END)
graph_builder.add_edge("normal_node",END)
graph=graph_builder.compile()

output = graph.invoke({
    "message": "hello langgraph"
})

print(output)