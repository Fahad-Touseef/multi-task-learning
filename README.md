# Sentence Transformers and Multi-Task Learning

This project implements:
1) a sentence transformer model
2) multi-task learning using a shared transformer backbone for sentence-level classification and token-level NER.

The implementation is CPU-based for now.

Refer to the [Explanation_for_Task_3_and_4.pdf](https://github.com/Fahad-Touseef/multi-task-learning/blob/main/Explanation_for_Task_3_and_4.pdf) for a brief write-up summarizing the key decisions and insights. Might require downloading for it to be displayed correctly.

## Docker Setup

### Requirements

Docker must be installed on your system (https://www.docker.com/products/docker-desktop)

### Build the Docker image

```bash
docker build -t sentence-transformer-mtl .
```

### Run the container

```bash
docker run -it sentence-transformer-mtl
```
