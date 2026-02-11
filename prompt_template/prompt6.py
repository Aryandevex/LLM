from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import json


import os

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# HF Conversational Endpoint
endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.1,
    max_new_tokens=200,
)

# Chat wrapper
chat_model = ChatHuggingFace(llm=endpoint)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an AI Query Router.

Classify the user query into exactly one intent:
search | summarize | code | explain

If ambiguous, choose "explain".

Output STRICT valid JSON only.

JSON format:
{{
  "intent": "<one of: search | summarize | code | explain>",
  "confidence": <float between 0 and 1>,
  "reason": "<short reasoning>"
}}

Do not output anything outside JSON.
"""
    ),
    (
        "human",
        "User Query:\n{query}"
    )
])

# LCEL chain
chain = prompt | chat_model

# 🔥 Example Query
result = chain.invoke({
    "query": "How does Redis help with Celery workers?"
})

print("Raw Model Output:")
print(result.content)

# Optional: Validate JSON safely
try:
    parsed = json.loads(result.content)
    print("\nParsed JSON:")
    print(parsed)
except Exception as e:
    print("\n⚠ Invalid JSON returned:", e)
