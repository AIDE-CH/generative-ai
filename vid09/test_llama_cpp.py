#%%

from llama_cpp import Llama

llm = Llama(
      model_path="./model/Llama-3.2-1B-Instruct-DOA_q8_0.gguf",
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
      verbose=False
)
output = llm(
      "Q: What is MUSIC for DOA (direction of arrival estimation) estimation? A: ", # Prompt
      max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(output["choices"][0]["text"])


#%%

from llama_cpp import Llama
llm = Llama(
      model_path="./model/Llama-3.2-1B-Instruct-DOA_q8_0.gguf",
      verbose=False
)
output = llm.create_chat_completion(
      messages = [
          {"role": "system", "content": "You are an assistant who perfectly answers question about algorithms for direction of arrival estimation (DOA)."},
          {
              "role": "user",
              "content": "What is MUSIC for DOA estimation?"
          }
      ]
)

print(output["choices"][0]["message"]["content"])