from mistralai import Mistral


def compute_embedding_for_text(client: Mistral, input) -> str:
    """
    Compute the text embedding using the Mistral API.
    """
    embeddings_batch_response = client.embeddings.create(
          model="mistral-embed",
          inputs=input
    )
    return embeddings_batch_response.data[0].embedding

