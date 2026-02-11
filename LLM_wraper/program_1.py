from transformers import pipeline

class LLMWrapper:
    def __init__(self, model_name="gpt2", task="text-generation", **pipeline_kwargs):
        self.pipe = pipeline(task, model=model_name, **pipeline_kwargs)
        print(f"Pipeline initialized for task: {task} with model: {model_name}")

    def generate_text(self, prompt, max_length=50, num_return_sequences=1):
        outputs = self.pipe(prompt, max_length=max_length, num_return_sequences=num_return_sequences)

        generated_texts = [output['generated_text'] for output in outputs]
        return generated_texts

llm = LLMWrapper(
    model_name="gpt2",
    task="text-generation"
)

result = llm.generate_text(
    prompt="Once upon a time in a futuristic city,",
    max_length=50
)

print(result)
