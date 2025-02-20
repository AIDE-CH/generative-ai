
from ollama import Client
client = Client(
  host='http://localhost:11434'
)

response = client.chat(
    model="llama3.2:1b",
    messages=[
    
        {
            "role": "user",
            "content": "why the sky is blue?",
        },
    ],
)
print(response["message"]["content"])