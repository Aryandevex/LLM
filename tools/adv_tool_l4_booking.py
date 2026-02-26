import os
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv, find_dotenv

# 1. Environment
load_dotenv(find_dotenv())
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.1
))

# 2. Level 4: Pydantic Schema for Tools
# This ensures the LLM provides specific, valid structured data.
class FlightBookingArgs(BaseModel):
    origin: str = Field(description="The departure city (e.g. NYC, Mumbai).")
    destination: str = Field(description="The arrival city.")
    passengers: int = Field(description="Number of people flying. Must be at least 1.")
    travel_date: str = Field(description="The date of travel in YYYY-MM-DD format.")

@tool(args_schema=FlightBookingArgs)
def book_flight(origin: str, destination: str, passengers: int, travel_date: str) -> str:
    """Books a flight in the system with full flight details."""
    return (
        f"FLIGHT BOOKED: {passengers} passenger(s) from {origin} to {destination} "
        f"on {travel_date}. Confirmation: FLY-{origin[:2].upper()}{destination[:2].upper()}"
    )

# 3. Setup Agent Logic
llm_with_tools = llm.bind_tools([book_flight])

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a travel agent. Extract all flight details before booking."),
    ("human", "{input}")
])

chain = prompt | llm_with_tools

def booking_bot(query: str):
    print(f"User: {query}")
    response = chain.invoke({"input": query})
    
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"--- Validating Booking Schema ---")
            # invoke() will raise a Pydantic ValidationError if the LLM provided bad data
            try:
                result = book_flight.invoke(tool_call["args"])
                print(f"Server Result: {result}")
            except Exception as e:
                print(f"Validation Error: {str(e)}")
    else:
        print(f"Bot: {response.content}")

if __name__ == "__main__":
    print("--- Level 4: Pydantic Validation ---")
    booking_bot("I want to fly from NYC up to Mumbai with 3 people on Christmas 2024.")
    booking_bot("Book a flight for 2 people tomorrow.") # Should fail or ask for details if LLM can't find them
