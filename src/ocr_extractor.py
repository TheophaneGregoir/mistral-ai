from mistralai import Mistral


def extract_text_from_file(client: Mistral, path: str) -> str:
    """
    Extract text from a file using the Mistral API.
    """
    # Upload the file to Mistral
    uploaded_file = client.files.upload(
        file={
            "file_name": path,
            "content": open(path, "rb"),
        },
        purpose="ocr"
    )  
    # Get the signed URL for the uploaded file
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id)

    # Use the signed URL to process the file with OCR
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        }
    )

    # return the extracted text
    return ocr_response.pages[0].markdown
