# Local RAG with MLX on Apple Silicon

This project provides a simple, end-to-end implementation of a Retrieval-Augmented Generation (RAG) pipeline that runs entirely locally on Apple Silicon hardware.

It uses a PDF document as a knowledge base, generates vector embeddings for its content, and leverages a local Large Language Model (LLM) via Apple's MLX framework to answer questions with context-aware responses. The entire process, from data processing to inference, happens offline on your machine, ensuring data privacy.

## Features

-   **PDF Parsing**: Extracts text from PDF documents page by page.
-   **Text Chunking**: Splits text into sentences and groups them into semantically relevant chunks.
-   **Vector Embeddings**: Creates numerical representations (embeddings) of text chunks using `sentence-transformers`.
-   **GPU-Accelerated**: Leverages Apple's Metal Performance Shaders (MPS) for fast embedding generation and model inference.
-   **Similarity Search**: Finds the most relevant context for a user's query using efficient vector similarity search.
-   **Local LLM Inference**: Generates answers using a local LLM (e.g., Gemma, Llama) running on the MLX framework, designed for Apple Silicon.
-   **End-to-End Pipeline**: A single script orchestrates the entire workflow from document ingestion to answer generation.

## Pipeline Workflow

The `data-processor.py` script executes the following steps:

1.  **Ingestion & Chunking**: The script reads a source PDF (`human-nutrition-text.pdf`), cleans the text, and splits it into manageable chunks of sentences.
2.  **Embedding & Storage**: Each chunk is converted into a vector embedding using the `all-mpnet-base-v2` model. These embeddings are saved to `text_chunks_and_embeddings_df.csv` to avoid reprocessing in the future.
3.  **Retrieval**: When a query is provided, the script embeds the query and uses cosine similarity to find the most relevant text chunks from the CSV file.
4.  **Augmentation & Generation**: The retrieved chunks (context) are combined with the original query into a detailed prompt. This augmented prompt is then fed to a local LLM running via MLX to generate a final, context-aware answer.

## Requirements

-   **Hardware**: An Apple Mac with an M-series chip (M1, M2, M3, etc.) is required to use MLX and MPS for GPU acceleration.
-   **Python**: 3.8 or newer.
-   **LLM**: A local copy of a Large Language Model that has been converted to the MLX format.

## Setup & Installation

1.  **Clone the Repository**
    
