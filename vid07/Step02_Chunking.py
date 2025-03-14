
#%%

"""
python -m pip install langchain langchain-community langchain_experimental chromadb

"""
import re
import os
import chromadb
from chromadb.config import Settings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings


btas = [95.0, 90, 80, 60, 30, 10, 1]
extra_token_length = 128
max_token_length = 512*1 + extra_token_length
SENTENCE_SPLITTER_REGEX = r"\s{2,}|[\n]+|[.?!]"
text_splitters_1 = [SemanticChunker(HuggingFaceEmbeddings(),
                                     breakpoint_threshold_type="percentile",
                                     breakpoint_threshold_amount=i,
                                     sentence_split_regex=SENTENCE_SPLITTER_REGEX,
                                     min_chunk_size=128)
                    for i in btas]

text_splitters_2 = [SemanticChunker(
    HuggingFaceEmbeddings(), breakpoint_threshold_type="gradient",
    breakpoint_threshold_amount=i,
    sentence_split_regex=SENTENCE_SPLITTER_REGEX,
    min_chunk_size=128
) for i in btas]
text_splitters = [item for pair in zip(text_splitters_1, text_splitters_2) for item in pair]



max_depth = len(text_splitters)
print(max_depth)


def splitAllChunkAtOnce(txt, depth=0):
    if txt == None:
        return []
    if len(txt) < (extra_token_length - 10):
        raise Exception("text is too small")
    if depth >= max_depth:
        print(txt)
        raise Exception("exceeded max depth")
    # use the semantic chunker at index = depth
    chunks = text_splitters[depth].split_text(txt)
    max_length = max(len(ic) for ic in chunks)
    if max_length > max_token_length:
        return splitAllChunkAtOnce(txt, depth=depth+1)
    else:
        print(f"split done depth={depth}")
        return (chunks, depth, chunkerUsed(text_splitters[depth]))


def chunkerUsed(c):
    name = type(c).__name__
    breakpoint_threshold_type = c.breakpoint_threshold_type
    breakpoint_threshold_amount = c.breakpoint_threshold_amount
    sentence_split_regex = c.sentence_split_regex
    min_chunk_size = c.min_chunk_size
    max_chunk_length = None
    
    return (name, breakpoint_threshold_type, breakpoint_threshold_amount, 
            sentence_split_regex, min_chunk_size, max_chunk_length)

# sometimes you need to add some kind of breaks to split on
'''
def appendNewLinesAfterArticle(text):
    # Regex pattern to find occurrences of **Article(number)**
    pattern = r'(\*\*Article\(\d+\)\*\*)'
    # Updated regex pattern to allow spaces around "Article" and inside parentheses
    pattern = r'(\*\*Article\s*\(\s*\d+\s*\)\*\*)'
    # Function to modify the match
    def append_text(match):
        return "\n" + match.group(1)
    # Replace and append
    new_text = re.sub(pattern, append_text, text)
    return new_text

'''

#%% chunking

md_files = [f for f in os.listdir("markdown/") if f.endswith(".md")]

chunked_docs = list()

for iii, mdf in enumerate(md_files):
    print("%d/%d (%s)"%(iii+1, len(md_files), mdf))
    
    text = None
    with open( "markdown/" + mdf, "r", encoding="utf-8") as out_file:
        text = out_file.read()
    
    if(text != None):
        chunks, depth, (name, breakpoint_threshold_type, breakpoint_threshold_amount, 
            sentence_split_regex, min_chunk_size, max_chunk_length) = splitAllChunkAtOnce(text, 0)
    
        chunked_docs.append({
            "title": mdf,
            "chunks": chunks,
            "depth": depth,
            "chunker":{
                "name": name,
                "breakpoint_threshold_type": breakpoint_threshold_type, 
                "breakpoint_threshold_amount": breakpoint_threshold_amount, 
                "sentence_split_regex": sentence_split_regex,
                "min_chunk_size": min_chunk_size,
                "max_chunk_length": max_chunk_length
                }
        })

#%% saving to chromadb
COLLECTION_NAME = "DIRECTION-FINDING-" + str( max_token_length - extra_token_length )
print (COLLECTION_NAME)

client = chromadb.PersistentClient("./db")
try:
    client.delete_collection(name=COLLECTION_NAME)
except Exception as ex:
    print(ex)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
print(collection.count())

for i, doc in enumerate(chunked_docs):
    for ii, c in enumerate(doc["chunks"]):
        collection.add(documents=[c],
                       metadatas=[{
                           "title": doc["title"],
                           "depth": doc["depth"],
                           "length": len(c),
                           "chunker-name": doc["chunker"]["name"],
                           "chunker-breakpoint_threshold_type": doc["chunker"]["breakpoint_threshold_type"],
                           "chunker-breakpoint_threshold_amount": doc["chunker"]["breakpoint_threshold_amount"],
                           "chunker-sentence_split_regex": doc["chunker"]["sentence_split_regex"],
                           "chunker-min_chunk_size": doc["chunker"]["min_chunk_size"],
                           "chunker-max_chunk_length": doc["chunker"]["max_chunk_length"] if doc["chunker"]["max_chunk_length"] else -1}],
                        ids=[f"{i+1}-{ii+1}"])
    
    print(collection.count())



