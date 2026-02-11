from transformers import pipeline
# Load sentiment analysis pipeline with a default model
pipe = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

result = pipe("This restaurant is awesome")

print(result)