from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


import os

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")



endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.0,
    max_new_tokens=120,
)


chat_model = ChatHuggingFace(llm=endpoint)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI that extracts user intent.\n"
        "You MUST respond with ONLY valid JSON.\n"
        "Do not add explanations, markdown, or extra text.\n\n"
        "JSON schema:\n"
        "{{\n"
        '  "intent": "<string>",\n'
        '  "confidence": "low | medium | high"\n'
        "}}"
    ),
    (
        "human",
        "User query: {query}"
    )
])

chain = prompt | chat_model


result = chain.invoke({
    "query": "I want to reset my account password."
})

print(result.content)
