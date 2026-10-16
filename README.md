# RAG-System
A Retrieval-Augmented Generation (RAG) system for restaurant data that uses embeddings and a vector database to retrieve relevant information and an LLM (Llama 3.2) to answer user questions based on the available data.

## Steps to Run the RAG System on Restaurant Data

### 1. Download LLM and Embedding Models
Open PowerShell and run:

```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
```

### 2. Setup Python Environment
Open a terminal (VS Code or any terminal) and run:

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

To upgrade pip (optional):

```bash
python -m pip install --upgrade pip
```

### 3. Create Vector Database
Run the following command to generate embeddings from restaurant data and store them in the vector database:

```bash
python vector.py
```

### 4. Run the RAG System
Run the following command:

```bash
python main.py
```

### 5. Output
Ask questions about the restaurant data, and the RAG system will retrieve relevant information and generate answers using the LLM.
