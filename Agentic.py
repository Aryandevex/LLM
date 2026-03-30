# Simulate an Agentic AI workflow

# Example goal
goal = "Fix login issue in my app"

# -----------------------------
# 1️⃣ Planner Agent
# -----------------------------
# Generate structured plan (normally from LLM)
plan = {
    "steps": [
        "Identify the error message during login",
        "Check authentication logs",
        "Verify backend login logic",
        "Inspect database",
        "Test login functionality"
    ]
}

print("\n[Planner Agent] Generated Plan:")
for i, step in enumerate(plan["steps"], 1):
    print(f"{i}. {step}")

# -----------------------------
# 2️⃣ Execution + Evaluation
# -----------------------------
execution_results = []

for i, step in enumerate(plan["steps"], 1):
    print(f"\n[Execution Agent] Performing Step {i}: {step}")

    # Simulate an issue if step mentions "database"
    if "database" in step.lower():
        result = "failed"
        print(f"[Evaluator Agent] Step {i} Result: {result} (issue found)")

        # -----------------------------
        # 3️⃣ Replanner Agent
        # -----------------------------
        print("[Replanner Agent] Adjusting plan...")
        new_step = "Check database connection and migrations"
        plan["steps"].insert(i, new_step)  # Insert new corrective step
        print(f"[Replanner Agent] New Step Added: {new_step}")

        # Assume the replanned step succeeds
        result = "success"
        print(f"[Evaluator Agent] Step {i} Result after replanning: {result}")
    else:
        result = "success"
        print(f"[Evaluator Agent] Step {i} Result: {result}")

    # Store results
    execution_results.append({"step": step, "result": result})

# -----------------------------
# 4️⃣ Memory Agent
# -----------------------------
memory_store = {
    "goal": goal,
    "plan": plan,
    "results": execution_results
}

print("\n[Memory Agent] Stored plan and results for future reference")

# -----------------------------
# 5️⃣ Show Final Plan and Results
# -----------------------------
print("\nFinal Plan After Execution and Replanning:")
for i, step in enumerate(plan["steps"], 1):
    print(f"{i}. {step}")

print("\nExecution Results:")
for r in execution_results:
    print(f"- {r['step']} → {r['result']}")
