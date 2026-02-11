"""Task: Simple classification / transformation
Skill learned: Writing clear instructions

Example: classify, summarize, rewrite
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace



# HF Endpoint (conversational model)
endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.1,
    max_new_tokens=50,
)
chat_model = ChatHuggingFace(llm=endpoint)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that summarize the text."),
    ("human", "Summarize the text.\n\nText: {text}")
])

chain = prompt | chat_model

result = chain.invoke({
    "text": "The product quality is good but delivery was slow."
})

print(result.content)