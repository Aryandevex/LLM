import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from serpapi import GoogleSearch

load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# ---------------- LLM ----------------
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        temperature=0.2,
        max_new_tokens=300
    )
)

# ---------------- TOOL ----------------
def search_tool(query):
    search = GoogleSearch({
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": 3
    })
    results = search.get_dict()
    
    snippets = []
    for r in results.get("organic_results", [])[:3]:
        snippets.append(r.get("snippet", ""))
    
    return "\n".join(snippets)

# ---------------- AGENT LOOP ----------------

def multi_step_agent(task, max_steps=8):
    memory = []
    context = ""
    
    for step in range(max_steps):
        print(f"\n🧠 Step {step+1}")

        # 1. THINK (decide next action)
        decision_prompt = f"""
You are an intelligent agent solving a task step-by-step.

TASK:
{task}

CURRENT CONTEXT:
{context}

MEMORY:
{memory}

Decide the next action. Choose ONE:
- THINK
- SEARCH
- WRITE
- VALIDATE
- FINAL

Respond ONLY like:
ACTION: <action>
INPUT: <what to do>
"""

        decision = llm.invoke(decision_prompt).content
        print("Decision:", decision)

        # Parse decision
        try:
            action = decision.split("ACTION:")[1].split("\n")[0].strip()
            action_input = decision.split("INPUT:")[1].strip()
        except:
            action = "THINK"
            action_input = ""

        # 2. ACT
        if action == "THINK":
            thought = llm.invoke(f"Think deeply about: {task}\nContext:{context}").content
            context += f"\nTHOUGHT: {thought}"
            memory.append(("THINK", thought))

        elif action == "SEARCH":
            result = search_tool(action_input)
            context += f"\nSEARCH RESULT: {result}"
            memory.append(("SEARCH", result))

        elif action == "WRITE":
            draft = llm.invoke(f"Write content for: {task}\nContext:{context}").content
            context += f"\nDRAFT: {draft}"
            memory.append(("WRITE", draft))

        elif action == "VALIDATE":
            validation = llm.invoke(f"""
Check if this is correct and complete:

{context}

Respond:
VALID: YES or NO
REASON:
""").content

            context += f"\nVALIDATION: {validation}"
            memory.append(("VALIDATE", validation))

            if "NO" in validation:
                fix = llm.invoke(f"Fix this:\n{context}").content
                context += f"\nFIXED: {fix}"
                memory.append(("FIX", fix))

        elif action == "FINAL":
            final = llm.invoke(f"Give final polished answer:\n{context}").content
            return final

        else:
            context += "\n[Unknown action, continuing...]"

    return "Max steps reached. Partial output:\n" + context


# ---------------- RUN ----------------
if __name__ == "__main__":
    task = "Create a structured article on Electric Vehicles with research-backed insights"
    result = multi_step_agent(task)
    print("\n\n🔥 FINAL OUTPUT:\n", result)
