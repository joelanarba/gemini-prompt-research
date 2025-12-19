from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Test call
response = client.chat.completions.create(
    model="google/gemini-flash-1.5",
    messages=[
        {"role": "user", "content": "Say 'Hello from OpenRouter!' if you can read this."}
    ],
)

print("✅ OpenRouter connection successful!")
print(f"Response: {response.choices[0].message.content}")
print(f"Model used: {response.model}")
print(f"Tokens: {response.usage.total_tokens}")