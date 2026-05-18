"""ETL step 5: validate outputs and populate Weaviate vector DB.

- Extract text from JECFA PDF monographs.
- Chunk with token-based R100-0 strategy (500-800 tokens, 100 overlap).
- Embed with sentence-transformers (nomic-embed-text).
- Upload to Weaviate and run a hybrid search query.
"""

from __future__ import annotations

import glob
import os
from typing import Iterable, List

import weaviate
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from weaviate.classes.config import Configure, DataType, Property

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "data", "raw", "03_jecfa_monographs")
WEAVIATE_URL = "http://localhost:8080"
CLASS_NAME = "ScientificMonograph"

MIN_TOKENS = 500
MAX_TOKENS = 800
OVERLAP = 100


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text:
            pages.append(text)
    return "\n".join(pages).strip()


def chunk_tokens(tokenizer, text: str) -> List[str]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    total = len(tokens)
    if total < MIN_TOKENS:
        return []
    if total <= MAX_TOKENS:
        return [tokenizer.decode(tokens)]

    chunks = []
    start = 0
    while start + MAX_TOKENS <= total:
        end = start + MAX_TOKENS
        chunks.append(tokenizer.decode(tokens[start:end]))
        start = end - OVERLAP

    remainder = total - start
    if remainder >= MIN_TOKENS:
        chunks.append(tokenizer.decode(tokens[start:]))
    else:
        print(
            f"Warning: dropping tail of {remainder} tokens (<{MIN_TOKENS}) for text length {total}."
        )

    return chunks


def iter_pdf_paths(pdf_dir: str) -> Iterable[str]:
    for path in sorted(glob.glob(os.path.join(pdf_dir, "*.pdf"))):
        yield path


def reset_schema(client) -> None:
    if client.collections.exists(CLASS_NAME):
        client.collections.delete(CLASS_NAME)
        print(f"Deleted existing class: {CLASS_NAME}")

    client.collections.create(
        name=CLASS_NAME,
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="source_filename", data_type=DataType.TEXT),
            Property(name="chemical_name", data_type=DataType.TEXT),
        ],
        vectorizer_config=Configure.Vectorizer.none(),
    )
    print(f"Created class: {CLASS_NAME}")


def main() -> None:
    client = weaviate.connect_to_local(port=8080, grpc_port=50051)
    try:
        if hasattr(client, "is_ready") and not client.is_ready():
            raise RuntimeError(f"Weaviate is not ready at {WEAVIATE_URL}")

        reset_schema(client)

        model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1", trust_remote_code=True
        )
        tokenizer = model.tokenizer

        objects = []
        for pdf_path in iter_pdf_paths(PDF_DIR):
            text = extract_text_from_pdf(pdf_path)
            if not text:
                print(f"Warning: no text extracted from {pdf_path}")
                continue

            chunks = chunk_tokens(tokenizer, text)
            if not chunks:
                print(f"Warning: text too short for chunking in {pdf_path}")
                continue

            filename = os.path.basename(pdf_path)
            chemical_name = os.path.splitext(filename)[0]

            embeddings = model.encode(chunks, batch_size=16, normalize_embeddings=True)
            for chunk, vector in zip(chunks, embeddings):
                objects.append(
                    {
                        "content": chunk,
                        "source_filename": filename,
                        "chemical_name": chemical_name,
                        "vector": vector.tolist(),
                    }
                )

        collection = client.collections.get(CLASS_NAME)
        with collection.batch.dynamic() as batch:
            for obj in objects:
                props = {
                    "content": obj["content"],
                    "source_filename": obj["source_filename"],
                    "chemical_name": obj["chemical_name"],
                }
                batch.add_object(properties=props, vector=obj["vector"])

        print(f"Uploaded {len(objects)} chunks to {CLASS_NAME}.")

        query_vector = model.encode(["Ethanol toxicity"], normalize_embeddings=True)[
            0
        ].tolist()
        result = collection.query.hybrid(
            query="Ethanol toxicity", vector=query_vector, limit=1
        )

        print("Top hybrid search result for 'Ethanol toxicity':")
        if result.objects:
            print(result.objects[0].properties)
        else:
            print(result)
    finally:
        client.close()


if __name__ == "__main__":
    main()
