import os
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv, find_dotenv

# 1. Setup
load_dotenv(find_dotenv())
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.1,
    max_new_tokens=200,
)
llm = ChatHuggingFace(llm=llm_endpoint)

# 2. Level 2 Tool: Weather (Simulated)
@tool
def fetch_weather(location: str):
    """Fetch the weather for a given city/location."""
    weather_data = {
        "london": "12°C, Rainy",
        "new york": "18°C, Cloudy",
        "mumbai": "32°C, Sunny"
    }
    return weather_data.get(location.lower(), f"Weather data for {location} not available.")

# 3. Assistant Logic
tools = [fetch_weather]
llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a weather specialist. Use tools to find real-time weather."),
    ("human", "{input}")
])

chain = prompt | llm_with_tools

def weather_bot(query: str):
    print(f"User: {query}")
    response = chain.invoke({"input": query})
    
    if response.tool_calls:
        for tool_call in response.tool_calls:
            # Concept: Manual selection of tool from a list
            if tool_call["name"] == "fetch_weather":
                res = fetch_weather.invoke(tool_call["args"])
                print(f"Tool (fetch_weather) result: {res}")
    else:
        print(f"Bot: {response.content}")

if __name__ == "__main__":
    print("--- Level 2: Weather Assistant ---")
    weather_bot("How is the weather in London?")
    weather_bot("What about Mumbai?")
