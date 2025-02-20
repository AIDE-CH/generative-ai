import ollama

for m in ollama.list().models:
    print(m.model + "    " + str(m.size/1e9))





