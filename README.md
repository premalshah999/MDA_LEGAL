# MDA Regulatory Chatbot Project

This repository provides a starting point for a Retrieval‑Augmented Generation (RAG) chatbot aimed at answering legal and regulatory questions about Maryland agriculture.  The project is split into two top‑level directories:

* `frontend` – A React/Vite application that implements the chat UI.  You should place your existing front‑end code in this folder.  If you already have a working front‑end from a prototype or another repository, copy it here.  The chatbot makes requests to the FastAPI back‑end described below.
* `backend` – A FastAPI service that provides API endpoints for ingestion and question answering.  It uses sentence‑transformer embeddings, a hybrid vector/sparse retriever and the Llama API to generate answers.  All of the core RAG logic lives in this folder.

At the root of the project you'll also find a `Dockerfile` that builds the back‑end into a container and a top‑level `README.md` (this file) describing how to run the system.

## Prerequisites

* **Python 3.10+** – The back‑end uses FastAPI, sentence‑transformers, FAISS and PyMuPDF.  A `requirements.txt` file is provided in `backend/`.
* **Node.js 16+** – Required only if you wish to build/run the front‑end.  See your front‑end framework’s documentation for exact version requirements.
* **Docker** – Optional but recommended.  The provided Dockerfile will build an image that exposes the FastAPI service on port 8000.

## Running the back‑end locally

1. Create a Python virtual environment and install dependencies:

   ```sh
   cd backend
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. Export your Llama API credentials.  The back‑end uses an OpenAI‑compatible API to call Llama models.  Set the following environment variables:

   ```sh
   export LLAMA_API_KEY=<your‑llama‑api‑key>
   export LLAMA_BASE_URL=https://api.llama.com/compat/v1/
   export LLAMA_MODEL=Llama-4-Maverick-17B-128E-Instruct-FP8  # or another available model
   ```

3. Place your regulatory documents in the `data/` directory at the project root.  The prototype supports both PDF and plain‑text files.  During start‑up the server will index everything in this folder.

4. Run the server:

   ```sh
   uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   Once started you can POST to `/ask` with JSON like `{ "question": "What are the rules for relocating a dwelling?", "k": 5 }` and receive a response containing an answer and citations.  The server will automatically re‑index on start‑up.  For larger deployments you may wish to move ingestion to a background job.

## Running with Docker

To build and run the back‑end using Docker:

```sh
docker build -t mda-rag-backend .
docker run -p 8000:8000 \
  -e LLAMA_API_KEY=<your‑llama‑api‑key> \
  -e LLAMA_BASE_URL=https://api.llama.com/compat/v1/ \
  -e LLAMA_MODEL=Llama-4-Maverick-17B-128E-Instruct-FP8 \
  -v $(pwd)/data:/app/data \
  mda-rag-backend
```

This will start the FastAPI server inside a container and mount your local `data/` directory into the container for ingestion.  Adjust environment variables as necessary for your Llama provider.

## Front‑end integration

The front‑end is expected to send POST requests to `/ask` on the back‑end.  A minimal example using `fetch` in a React component might look like this:

```ts
const askQuestion = async (question: string) => {
  const res = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, k: 5 }),
  });
  const data = await res.json();
  // data.answer contains the answer and citations
  console.log(data.answer, data.sources);
};
```

For development, you can set up a proxy in Vite or Next.js to forward `/ask` to `http://localhost:8000/ask`.

## Updating the corpus

This prototype indexes all documents on start‑up.  To add or update documents:

1. Place new PDFs or text files into the `data/` directory.
2. Restart the back‑end process (or redeploy the Docker container).

The ingestion logic automatically chunks PDFs, computes sentence‑transformer embeddings and builds a hybrid vector/sparse index.  You can modify `backend/app/rag.py` to adjust chunk sizes, overlap or retrieval strategy.

## Scalability

The architecture of this prototype is designed to scale.  The current implementation keeps the vector index and TF–IDF matrix in memory for simplicity.  In a production deployment you should:

* Replace the in‑memory FAISS index with [pgvector](https://github.com/pgvector/pgvector) or a managed vector database like Pinecone or Weaviate.
* Serve the model inference via a GPU‑enabled service if your Llama provider supports it or host your own quantized model behind an OpenAI‑compatible API.
* Run ingestion as a background job (e.g. Celery, RQ, or built‑in FastAPI background tasks) so that new documents can be added without downtime.
* Introduce caching (Redis) and request coalescing to reduce LLM calls for repeated queries.

With these adjustments the system should comfortably handle hundreds or thousands of concurrent users.
