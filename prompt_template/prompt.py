
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace



#  HF Endpoint (conversational model)
endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.1,
    max_new_tokens=50,
)

# Chat wrapper (THIS IS THE KEY)
chat_model = ChatHuggingFace(llm=endpoint)

#  Chat Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that classifies sentiment."),
    ("human", "Classify the sentiment as Positive, Negative, or Neutral.\n\nText: {text}")
])

# LCEL chain
chain = prompt | chat_model

# Invoke
result = chain.invoke({
    "text": "The product quality is good but delivery was slow."
})

print(result.content)
