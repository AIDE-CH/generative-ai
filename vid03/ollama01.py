
import ollama

response = ollama.chat(
    model="llama3.2:1b",
    messages=[
        {
            "role": "user",
            "content": "Tell me an interesting fact about LLMs and be lengthy!",
        },
    ],
    stream=True
)

for chunk in response:
    print(chunk["message"]["content"], end="")

