import os
from mistralai import Mistral

# Set up your environment variable for the Mistral API key
# Read the key from the text file free_account_key.txt

with open("free_account_key.txt", "r") as file:
    api_key = file.read().strip()

model = "mistral-small-latest"

client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
    model= model,
    messages = [
        {
            "role": "user",
            "content": "What is the best French cheese?",
        },
    ]
)

print(chat_response.choices[0].message.content)