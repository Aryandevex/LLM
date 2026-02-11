from typing import TypedDict

class state(TypedDict):
    message: str
    status: str
def check_node(state: state):
    if state["message"].lower() == "yes":
        return {"status": "approved"}
    return {"status": "retry"}
def retry_node(state: state):
    print("Not approved, trying again...")
    return state
def loop_router(state: state):
    if state["status"] == "approved":
        return "end"
    return "retry"

from langgraph.graph import StateGraph

graph = StateGraph(state)

graph.add_node("check", check_node)
graph.add_node("retry", retry_node)

graph.add_conditional_edges(
    "check",
    loop_router,
    {
        "retry": "retry",
        "end": "__end__"
    }
)

graph.add_edge("retry", "check")

graph.set_entry_point("check")

app = graph.compile()
output = app.invoke({
    "message": "yes"
})

print(output)


