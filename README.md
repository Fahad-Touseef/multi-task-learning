# Sentence Transformers and Multi-Task Learning

This project implements:
1) a sentence transformer model
2) multi-task learning using a shared transformer backbone for sentence-level classification and token-level NER.
The implementation is CPU-based for now.

## Docker Setup

### Requirements

- Docker installed on your system (https://www.docker.com/products/docker-desktop)

### Build the Docker image

```bash
docker build -t sentence-transformer-mtl .
```

### Run the container

```bash
docker run -it sentence-transformer-mtl
```