# RetServe - Retrieval Service for Lab Internal Use

## Overview
RetServe is a simple yet efficient retrieval service that handles text embedding, vector indexing, and search functionality using FAISS and embedding models.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

All configurations are managed through YAML files in the `config/` directory:

- `config/embed.yaml`: Embedding model and data paths
- `config/index.yaml`: Index building parameters
- `config/serve.yaml`: Service deployment settings
- `config/log.yaml`: Logging configuration

Edit these files to modify paths, model settings, and other parameters as needed.

## Usage

### 1. Embed Corpus
Generate embeddings for your text corpus:

```bash
python embed.py
```

Options:
- `--config <name>`: Use custom configuration file (default: embed)

### 2. Build Index
Create FAISS index from embeddings:

```bash
python index.py
```

Options:
- `--config <name>`: Use custom configuration file (default: index)

### 3. Start Retrieval Service
Launch the HTTP API service:

```bash
python ret_serve.py
```

Options:
- `--config <name>`: Use custom configuration file (default: serve)

## API Endpoints

- `GET /health`: Health check endpoint
- `POST /search`: Search endpoint
  - Request body: `{"queries": ["query1", "query2"], "topk": 5}`
  - Response: `{"contents": [[{"id": "1", "title": "...", "text": "...", "contents": "..."}]], "scores": [[0.98, 0.95, ...]]}`

- API documentation available at `/docs` (Swagger UI) and `/redoc`

## Project Structure

```
RetServe/
├── src/                  # Source code
│   ├── embed.py          # Embedding implementation
│   ├── index.py          # Indexing implementation
│   ├── ret_serve.py      # Service implementation
│   ├── logging.py        # Logging setup
│   └── config_loader.py  # Configuration loader
├── config/               # Configuration files
├── data/                 # Data storage
├── logs/                 # Log files
├── embed.py              # Embedding entry point
├── index.py              # Indexing entry point
├── ret_serve.py          # Service entry point
└── requirements.txt      # Dependencies
```

## Notes for Internal Use

1. Make sure to update configuration files with correct paths for your environment
2. The service is intended for internal lab use only
3. Default settings are optimized for the provided example data
4. For large datasets, consider adjusting batch sizes and chunk sizes
5. Logs are stored in the `logs/` directory

## Troubleshooting

- If you encounter GPU-related issues, check your CUDA environment and GPU IDs in configuration files
- Ensure all dependencies are properly installed in the `ur-hx` conda environment
- Verify that input files exist at the paths specified in configuration files
- Check log files for detailed error messages
