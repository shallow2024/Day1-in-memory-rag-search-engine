"""Day 1: an in-memory retrieval-augmented generation search engine."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Sequence

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from groq import Groq


MODEL_NAME = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "technical_docs"
DEFAULT_MAX_DISTANCE = 0.75
INSUFFICIENT_CONTEXT_MESSAGE = "The retrieved context does not contain enough information."


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    title: str
    text: str


DOCUMENTS: tuple[DocumentChunk, ...] = (
    DocumentChunk(
        "doc-1",
        "Retrieval-Augmented Generation",
        "RAG combines document retrieval with language generation. The application first retrieves relevant context, then asks a language model to answer using that context. This reduces unsupported answers when the model has access to a trusted knowledge base.",
    ),
    DocumentChunk(
        "doc-2",
        "ChromaDB In-Memory Client",
        "ChromaDB can create an ephemeral in-memory client with chromadb.Client(). Data exists only for the lifetime of the Python process. This mode is useful for demos, tests, and short-lived experiments because it does not create a database directory.",
    ),
    DocumentChunk(
        "doc-3",
        "Sentence Embeddings",
        "Sentence-transformers convert text into dense numerical vectors called embeddings. Semantically similar sentences tend to have nearby vectors, allowing an application to retrieve relevant text even when the query does not use exactly the same words as the document.",
    ),
    DocumentChunk(
        "doc-4",
        "FastAPI Service Design",
        "A FastAPI application can expose a POST endpoint that accepts a question and returns an answer. Input validation, timeouts, structured errors, and logs should be added before exposing an AI service to other applications.",
    ),
    DocumentChunk(
        "doc-5",
        "Grounded Answers",
        "A grounded RAG answer should use only the retrieved context. If the context does not contain the answer, the assistant should say that the documents do not provide enough information rather than inventing a response.",
    ),
)


class MissingApiKeyError(RuntimeError):
    """Raised when generation is requested without a Groq API key."""


def build_collection() -> Collection:
    """Create and seed a fresh, ephemeral ChromaDB collection."""
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.Client()
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"description": "Day 1 technical documentation"},
    )
    collection.add(
        ids=[document.chunk_id for document in DOCUMENTS],
        documents=[document.text for document in DOCUMENTS],
        metadatas=[{"title": document.title} for document in DOCUMENTS],
    )
    return collection


def retrieve(collection: Collection, query: str, top_k: int = 3) -> list[dict[str, str]]:
    """Retrieve the most semantically relevant document chunks."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    result = collection.query(query_texts=[query], n_results=min(top_k, len(DOCUMENTS)))
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return [
        {
            "title": str(metadata.get("title", "Untitled")),
            "text": str(document),
            "distance": f"{float(distance):.4f}",
        }
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]


def has_relevant_context(
    context: Sequence[dict[str, str]], max_distance: float = DEFAULT_MAX_DISTANCE
) -> bool:
    """Return whether the closest chunk is relevant enough to use."""
    if max_distance <= 0:
        raise ValueError("max_distance must be greater than 0.")
    return bool(context) and float(context[0]["distance"]) <= max_distance


def build_prompt(query: str, context: Sequence[dict[str, str]]) -> str:
    """Build a strict prompt that keeps the generation grounded."""
    context_text = "\n\n".join(
        f"[{item['title']}]\n{item['text']}" for item in context
    )
    return (
        "You are a technical documentation assistant. Answer the question using only "
        "the context below. If the context is insufficient, say exactly that the "
        "documentation does not provide enough information. Do not invent details. "
        "Keep the answer concise and mention the relevant document title.\n\n"
        f"CONTEXT:\n{context_text}\n\nQUESTION:\n{query}"
    )


def generate_answer(query: str, context: Sequence[dict[str, str]]) -> str:
    """Generate a grounded answer through the Groq SDK."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise MissingApiKeyError(
            "GROQ_API_KEY is missing. Set it in your environment to enable generation. "
            "Retrieval can still be inspected with --retrieve-only."
        )

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.1,
        max_tokens=350,
        messages=[
            {
                "role": "system",
                "content": "Answer only from supplied technical documentation.",
            },
            {"role": "user", "content": build_prompt(query, context)},
        ],
    )
    answer = completion.choices[0].message.content
    if not answer:
        raise RuntimeError("Groq returned an empty response.")
    return answer.strip()


def print_retrieval(context: Sequence[dict[str, str]]) -> None:
    print("\nRetrieved context:")
    for index, item in enumerate(context, start=1):
        print(f"\n{index}. {item['title']} (distance: {item['distance']})")
        print(f"   {item['text']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query an in-memory ChromaDB RAG engine backed by Groq."
    )
    parser.add_argument("query", nargs="?", help="Question to ask the technical docs")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument(
        "--max-distance",
        type=float,
        default=DEFAULT_MAX_DISTANCE,
        help=f"Reject weak matches above this Chroma distance (default: {DEFAULT_MAX_DISTANCE})",
    )
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Print retrieved chunks without calling Groq",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    query = args.query or input("Ask a question about the technical docs: ").strip()
    if not query:
        print("Error: query cannot be empty.", file=sys.stderr)
        return 2

    try:
        collection = build_collection()
        context = retrieve(collection, query, top_k=args.top_k)
        if not has_relevant_context(context, max_distance=args.max_distance):
            print(f"\n{INSUFFICIENT_CONTEXT_MESSAGE}")
            return 0
        print_retrieval(context)
        if args.retrieve_only:
            return 0
        print("\nGrounded answer:")
        print(generate_answer(query, context))
        return 0
    except MissingApiKeyError as error:
        print(f"\nGeneration unavailable: {error}", file=sys.stderr)
        return 3
    except (ValueError, RuntimeError) as error:
        print(f"\nError: {error}", file=sys.stderr)
        return 1
    except Exception as error:  # Keep the CLI readable for SDK/network failures.
        print(f"\nUnexpected error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
