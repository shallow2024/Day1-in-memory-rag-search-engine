# Day1 — In-Memory RAG Search Engine

A small, transparent Retrieval-Augmented Generation (RAG) prototype built in one day. It seeds five technical documentation chunks into an **ephemeral ChromaDB collection**, retrieves the most semantically relevant chunks with **Sentence Transformers**, and optionally asks Groq's `llama-3.3-70b-versatile` to generate a grounded answer.

This project intentionally uses no local database files. The vector store is recreated in memory every time the script starts.

## Architecture

```mermaid
flowchart LR
    A[User question] --> B[Sentence Transformers\nall-MiniLM-L6-v2]
    B --> C[ChromaDB\nIn-memory collection]
    C --> D[Top-k technical chunks]
    D --> E[Grounded prompt]
    E --> F[Groq API\nllama-3.3-70b-versatile]
    F --> G[Answer with source title]
```

## What this demonstrates

- The basic RAG pattern: **retrieve first, generate second**.
- Semantic retrieval rather than exact keyword matching.
- In-memory ChromaDB for a zero-database-file prototype.
- Prompt grounding: the generator is instructed to use only retrieved context.
- Safe failure when `GROQ_API_KEY` is missing.
- A `--retrieve-only` mode that works without a Groq key.

## Requirements

- Python 3.10 or newer.
- A Groq API key for generated answers. The key is optional if you only want to inspect retrieval.
- Internet access the first time Sentence Transformers downloads `all-MiniLM-L6-v2`.

Groq's API may have free developer access and usage limits. Check the current Groq account terms before use. Never commit your API key.

## Windows setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the environment's executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run retrieval without a Groq key

This confirms that the local embedding model and in-memory vector database work:

```powershell
python rag_engine.py "How does ChromaDB store data in this demo?" --retrieve-only
```

Expected behavior: the script prints the three most relevant technical chunks and exits successfully.

## Run with Groq generation

PowerShell for the current terminal session:

```powershell
$env:GROQ_API_KEY = "paste-your-own-key-here"
python rag_engine.py "How does RAG reduce unsupported answers?"
```

Or with the environment's executable:

```powershell
$env:GROQ_API_KEY = "paste-your-own-key-here"
.\.venv\Scripts\python.exe rag_engine.py "What happens if the documentation does not contain an answer?"
```

The script prints:

1. Retrieved chunks and their distances.
2. A grounded answer from Groq.

To clear the key when finished:

```powershell
Remove-Item Env:GROQ_API_KEY
```

## CLI options

```text
python rag_engine.py [query] [--top-k 3] [--retrieve-only]
```

Examples:

```powershell
python rag_engine.py "What are sentence embeddings?" --top-k 2 --retrieve-only
python rag_engine.py "How should a FastAPI AI service handle errors?"
```

## Data lifecycle

The five documentation chunks are constants inside `rag_engine.py`. On each run:

```text
Python process starts
→ ChromaDB creates an ephemeral client
→ five chunks are embedded and inserted
→ query retrieves relevant chunks
→ process exits
→ vectors disappear
```

This is deliberately not production persistence. A later project can replace `chromadb.Client()` with a persistent or hosted database.

## Safety and limitations

- This is a learning prototype, not a production support system.
- The answer is only as reliable as the retrieved context and model response.
- The prompt requests grounded answers, but generation should still be reviewed for important use cases.
- Do not place secrets or private customer documents in this repository.
- Groq calls send the prompt and retrieved context to an external service; review the provider's current privacy and retention terms before using sensitive data.

## License

MIT. The technical documentation chunks are original educational examples created for this project.
