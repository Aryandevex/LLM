"""
A real agent is defined by 5 characteristics:

Goal-Driven – It has a purpose it is trying to achieve.
Planning / Reasoning – It thinks about steps to reach the goal.
Action / Execution – It performs tasks or interacts with the environment.
Observation / Feedback – It checks results and knows success or failure.
Adaptation / Memory – It changes behavior if the first attempt fails and remembers past experience.
"""
goal = "Fix login issue in my app"

# 1. Planner thinks
plan = generate_plan(goal)
print("\n[Planner Agent] Generated Plan:")
for i, step in enumerate(plan["steps"], 1):
    print(f"{i}. {step}")

# 2. Simulate Execution and Observation
execution_results = []
for i, step in enumerate(plan["steps"], 1):
    print(f"\n[Execution Agent] Performing Step {i}: {step}")
    # Here you simulate success/failure
    if "database" in step.lower():
        result = "failed"
        print(f"[Evaluator Agent] Step {i} Result: {result} (issue found)")
        # 3. Adaptive replanning
        print("[Replanner Agent] Adjusting plan...")
        plan["steps"].insert(i, "Check DB connection and migrations")
        print("[Replanner Agent] New Step Added")
        result = "success"
    else:
        result = "success"
        print(f"[Evaluator Agent] Step {i} Result: {result}")

    execution_results.append({"step": step, "result": result})

# 4. Memory Agent stores plan
memory_store = {"goal": goal, "plan": plan, "results": execution_results}
print("\n[Memory Agent] Stored plan and results for future reference")
