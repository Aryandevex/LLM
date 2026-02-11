from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import json


import os

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# HF Endpoint (conversational model)
endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.1,
    max_new_tokens=400,
)

chat_model = ChatHuggingFace(llm=endpoint)

# 🔥 PRO SYSTEM PROMPT
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a senior backend architecture reviewer.

Analyze the given backend system description carefully.

Identify:
- scalability_issues
- security_risks
- performance_bottlenecks
- recommended_improvements
- risk_score (integer from 1 to 10)

Rules:
- Only analyze what is explicitly mentioned.
- Do NOT hallucinate missing technologies.
- If something is not mentioned, do not assume it exists.
- Be precise and technical.
- Output STRICT valid JSON only.
- No markdown.
- No explanation outside JSON.

JSON format:
{{
  "scalability_issues": ["string"],
  "security_risks": ["string"],
  "performance_bottlenecks": ["string"],
  "recommended_improvements": ["string"],
  "risk_score": 0
}}

Before responding:
- Ensure output is valid JSON.
- Ensure risk_score is integer between 1 and 10.
- Ensure arrays contain strings only.
"""
    ),
    (
        "human",
        "System Description:\n{architecture}"
    )
])

chain = prompt | chat_model

# 🧨 Example Architecture Input
architecture_input = """
We have a Django app deployed on a single EC2 instance.
Celery runs on the same machine.
Redis is used as broker.
No load balancer.
No autoscaling.
Database is SQLite.
"""

result = chain.invoke({
    "architecture": architecture_input
})

print("Raw Model Output:\n")
print(result.content)

# 🔍 Optional JSON Validation
try:
    parsed = json.loads(result.content)
    print("\nParsed JSON:\n")
    print(parsed)
except Exception as e:
    print("\n⚠ Invalid JSON returned:", e)
