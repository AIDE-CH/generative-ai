
#%%
import chromadb
import ollama


COLLECTION_NAME = "DIRECTION-FINDING-512"
db_client = chromadb.PersistentClient("./db")
db_collection = db_client.get_collection(name=COLLECTION_NAME)


def query(question, n_results=10):
    results = db_collection.query(query_texts=[question], n_results=n_results)
    ids = results['ids'][0]
    distances = results['distances'][0]
    documents =  results['documents'][0]
    metadatas = results['metadatas'][0]
    return (ids, documents, metadatas, distances)


#%% first question without context

question = "What is the ESPRIT algorithm for direction of arrival estimation"

options = {
    'temperature': 0.0
}

response = ollama.chat(
    model="llama3.2:1b",
    messages=[
        {
            "role": "user",
            "content": question
        },
    ],
    stream=True,
    options=options
)

for chunk in response:
    print(chunk["message"]["content"], end="")


#%%

ids, documents, metadatas, distances = query(question, 10)

print(distances)
print(metadatas[0])
print(documents[0])


question_with_context = "You are a helpful assistant. You answer questions based only on the given context"+\
                        "GIVEN CONTEXT:\n" + '\n'.join(documents) + \
                        "\n\n" + "Question: " + question + "\n" +\
                        "Answer: "

response = ollama.generate(
    model="llama3.2:1b",
    prompt=question_with_context,
    stream=True,
    options=options
)

print("\n\n\n")
for chunk in response:
    print(chunk["response"], end="")



#%% augmented query

"""
- here we augment a reformulation of the question.
- some also try to use the answer as augmentation. 
   This will work if the LLM already know an approximate answer to the question
"""
response = ollama.generate(
    model="llama3.2:1b",
    prompt="Rephrase the following question. Question: " + question + "\n Rephrased question: ",
    stream=False,
    options=options
)

response = response["response"]
print(response)

#%%


ids, documents, metadatas, distances = query(question + "\n" + response, 10)

print(distances)
print(metadatas[0])
print(documents[0])


question_with_context = "You are a helpful assistant. You answer questions based only on the given context"+\
                        "GIVEN CONTEXT:\n" + '\n'.join(documents) + \
                        "\n\n" + "Question: " + question + "\n" +\
                        "Answer: "

response = ollama.generate(
    model="llama3.2:1b",
    prompt=question_with_context,
    stream=True
)

print("\n\n\n")
for chunk in response:
    print(chunk["response"], end="")