import os
import faiss
from mistralai import Mistral
from text_embedder import compute_embedding_for_text
import numpy as np
import json

# Set up your environment variable for the Mistral API key
# Read the key from the text file free_account_key.txt

with open("small_deposit_key.txt", "r") as file:
    api_key = file.read().strip()

client = Mistral(api_key=api_key)

question = "I want find the best candidate to be a male actor in a musical. Can you help me?"
question_embeddings = np.array([compute_embedding_for_text(client, question)])

# Load the text extracts from the JSON file
with open("data/db/text_extracts.json", "r") as f:
    text_extracts = json.load(f)

#Load the FAISS index from the file created in build_vector_db.py
index = faiss.read_index("data/db/vector_db.index")

# Perform the search
D, I = index.search(question_embeddings, k=2) # distance, index
retrieved_text = [text_extracts[i] for i in I.tolist()[0]]


prompt = f"""
Context information is below.
---------------------
{retrieved_text}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:
"""

# Call the Mistral API to get the answermessages = [
messages = [
    {
        "role": "user", "content": prompt
    }
]
chat_response = client.chat.complete(
    model='mistral-small-latest',
    messages=messages
)

# Print the answer
print(chat_response.choices[0].message.content)