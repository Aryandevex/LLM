from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


endpoint=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.1,
    max_new_tokens=50,
)
chat_model = ChatHuggingFace(llm=endpoint)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a {role}. Answer the question from this perspective."
    ),
    (
        "human",
        "Question: {question}"
    )
])

chain = prompt | chat_model

question = "What is an API?"

roles = [
    "senior backend engineer",
    "product manager",
    "beginner-friendly programming teacher"
]

for role in roles:
    result = chain.invoke({
        "role": role,
        "question": question
    })
    print(f"\nRole: {role}")
    print(result.content)