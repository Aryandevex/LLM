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
        "You are an assistant that answers questions using ONLY the provided context. "
        "If the answer is not present in the context, reply exactly with: "
        "'Not found in context'."
    ),
    (
        "human",
        "Context:\n{context}\n\n"
        "Question:\n{question}"
    )
])

chain = prompt | chat_model

# 5️⃣ Invoke (answer exists)
result_1 = chain.invoke({
    "context": (
        "LangChain is a Python framework used to build applications "
        "powered by large language models."
    ),
    "question": "What is LangChain?"
})

print("Answer 1:", result_1.content)


result_2 = chain.invoke({
    "context": (
        "LangChain is a Python framework used to build applications "
        "powered by large language models."
    ),
    "question": "Who created LangChain?"
})

print("Answer 2:", result_2.content)