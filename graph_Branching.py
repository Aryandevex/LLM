# program_2_branching_graph.py
"""
A graph that routes to different nodes based on input content.
Like a customer service bot that directs queries to different departments.
"""

from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
import random

class State(TypedDict):
    query: str
    response: str
    sentiment: str

# === NODES ===
def analyze_sentiment(state: State):
    """Detect if query is positive, negative, or neutral"""
    query = state['query'].lower()
    
    if any(word in query for word in ['terrible', 'bad', 'awful', 'hate']):
        sentiment = "negative"
    elif any(word in query for word in ['great', 'awesome', 'love', 'good']):
        sentiment = "positive"
    else:
        sentiment = "neutral"
    
    print(f"🔍 Sentiment analysis: {sentiment}")
    return {"sentiment": sentiment}

def handle_positive(state: State):
    response = f"🎉 Thanks for your positive feedback! We're glad you're happy with: {state['query']}"
    return {"response": response}

def handle_negative(state: State):
    response = f"😟 We're sorry to hear that. Our team will contact you about: {state['query']}"
    return {"response": response}

def handle_neutral(state: State):
    response = f"🤖 Let me help you with that: {state['query']}"
    return {"response": response}

def escalate_unknown(state: State):
    response = f"🚨 Escalating to human agent: {state['query']}"
    return {"response": response}

# === ROUTER (Conditional Edge) ===
def router(state: State) -> Literal["handle_positive", "handle_negative", "handle_neutral", "escalate_unknown"]:
    """Decide which node to go to based on sentiment"""
    sentiment = state.get("sentiment", "neutral")
    
    if sentiment == "positive":
        return "handle_positive"
    elif sentiment == "negative":
        return "handle_negative"
    elif sentiment == "neutral":
        return "handle_neutral"
    else:
        return "escalate_unknown"

# === BUILD GRAPH ===
builder = StateGraph(State)

# Add all nodes
builder.add_node("analyze", analyze_sentiment)
builder.add_node("handle_positive", handle_positive)
builder.add_node("handle_negative", handle_negative)
builder.add_node("handle_neutral", handle_neutral)
builder.add_node("escalate_unknown", escalate_unknown)

# Add edges
builder.add_edge(START, "analyze")
builder.add_conditional_edges("analyze", router)  # Magic happens here!
builder.add_edge("handle_positive", END)
builder.add_edge("handle_negative", END)
builder.add_edge("handle_neutral", END)
builder.add_edge("escalate_unknown", END)

graph = builder.compile()

# === TEST DIFFERENT INPUTS ===
test_queries = [
    "I love your product!",
    "This is terrible service",
    "What are your hours?",
    "Something completely random"
]

print("\n" + "="*60)
for query in test_queries:
    print(f"\n📝 Query: {query}")
    result = graph.invoke({"query": query, "response": "", "sentiment": ""})
    print(f"💬 Response: {result['response']}")
    print("-"*40)

"""
EXPECTED: 
- Positive queries get celebration response
- Negative get apology + escalation
- Neutral get helpful response
"""
