import os
import traceback
from langchain_community.llms import HuggingFaceEndpoint

# 🔑 Set your Hugging Face API Key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_huggingface_api_key"


class SelfHealingAgent:

    def __init__(self):
        print("\n[Step 1] Initializing Qwen model from Hugging Face...")

        self.llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            temperature=0.2,
            max_new_tokens=512,
        )

    # Step 2: Run code
    def run_code(self, code):
        try:
            exec(code)
            return "success", None, None
        except Exception as e:
            return "error", str(e), traceback.format_exc()

    # Step 3: Ask AI to fix code
    def fix_code(self, code, error, trace):
        print("\n🤖 Asking Qwen to fix the code...")

        prompt = f"""
You are a Python expert.

Fix the following code.
Return ONLY corrected code. No explanation.

Code:
{code}

Error:
{error}

Traceback:
{trace}
"""

        response = self.llm.invoke(prompt)

        return response.strip()

    # Step 4: Agent loop
    def start(self, code):
        for i in range(3):
            print(f"\n🚀 Attempt {i+1}")
            print("\nRunning Code:\n", code)

            status, error, trace = self.run_code(code)

            if status == "success":
                print("\n✅ Code executed successfully!")
                return code

            print("\n❌ Error:", error)

            code = self.fix_code(code, error, trace)

        print("\n⚠️ Failed to fix after 3 attempts.")
        return None


# -------------------------
# 🚀 Run the Agent
# -------------------------

if __name__ == "__main__":

    buggy_code = """
pritn("Hello World")

x = 10 / 0
print(y)
"""

    agent = SelfHealingAgent()
    fixed_code = agent.start(buggy_code)

    print("\n📌 Final Fixed Code:\n", fixed_code)
