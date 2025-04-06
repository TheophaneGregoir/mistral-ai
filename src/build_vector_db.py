import os
import argparse
import faiss
from mistralai import Mistral
from ocr_extractor import extract_text_from_file
from text_embedder import compute_embedding_for_text
import numpy as np
import json

# Define the parser
parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('--run_ocr', action="store", dest='run_ocr', default="no_run", type=str)
parser.add_argument('--run_embedding', action="store", dest='run_embedding', default="no_run", type=str)

args = parser.parse_args()


# Set up your environment variable for the Mistral API key
# Read the key from the text file free_account_key.txt
with open("small_deposit_key.txt", "r") as file:
    api_key = file.read().strip()

client = Mistral(api_key=api_key)

pdf_path_list = ["data/pdf/" + file for file in os.listdir("data/pdf")]

# OCR Extraction of text from PDF
if args.run_ocr == "run":
    text_extracts = []
    for pdf_path in pdf_path_list:
        # Extract text from the PDF
        text_extracts.append(extract_text_from_file(client, pdf_path))
    # Save the text extracts to a JSON file
    with open("data/db/text_extracts.json", "w") as f:
        json.dump(text_extracts, f)
else:
    # Load the text extracts from the JSON file
    with open("data/db/text_extracts.json", "r") as f:
        text_extracts = json.load(f)
    print("Length of the extracted texts: " + str(len(text_extracts)))

if args.run_embedding == "run":
    # Compute text embeddings
    text_embeddings = np.array([compute_embedding_for_text(client, text) for text in text_extracts])

    # Load all these files into a FAISS vector database
    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)

    # Save the index to a file
    faiss.write_index(index, "data/db/vector_db.index")
else:
    # Load the index from a file
    index = faiss.read_index("data/db/vector_db.index")
