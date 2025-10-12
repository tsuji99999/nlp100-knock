import os
from openai import OpenAI

class LLMClient:
    def __init__(self, model_name="gpt-5-mini"):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def ask(self, prompt):
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt
        )
        return response