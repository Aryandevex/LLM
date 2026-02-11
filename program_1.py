from typing import TypedDict
class state(TypedDict):
    message:str
    result:str

def process_message(state:state):
    print("node is runing")
    return {
        "result":state["message"].upper()
    }
from langgraph.graph import StateGraph,END,START
graph_builder=StateGraph(state)
graph_builder.add_node("process_message",process_message)
graph_builder.add_edge(START,"process_message")
graph_builder.add_edge("process_message",END)
graph=graph_builder.compile()

output = graph.invoke({
    "message": "hello langgraph"
})

print(output)