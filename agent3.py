import os
from langchain_community.llms import HuggingFaceEndpoint

# 🔑 API Key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_huggingface_api_key"


class CodeReviewAgent:

    def __init__(self):
        print("\n[Reviewer Agent] Initializing...")

        self.llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            temperature=0.3,
            max_new_tokens=300,
        )

    # Step 1: Review code using AI
    def review_code(self, code):
        print("\n🤖 Reviewing code...")

        prompt = f"""
You are a senior Python developer.

Review the following code and:
- Find mistakes
- Suggest improvements
- Suggest better practices

Code:
{code}
"""

        response = self.llm.invoke(prompt)
        return response

    # Step 2: Agent loop (simple version)
    def start(self, code):
        print("\n📌 Code to review:\n", code)

        review = self.review_code(code)

        print("\n📝 Review Result:\n")
        print(review)


# -------------------------
# 🚀 Run Agent
# -------------------------

if __name__ == "__main__":

    sample_code = """
def add(a,b):
 return a+b

print(add(2,3))
"""

    agent = CodeReviewAgent()
    agent.start(sample_code)
