# E2E-lane-detection

# 🚗 End-to-End ML Pipeline for Autonomous Driving

This repository contains a complete ML lifecycle for an autonomous driving perception task — lane detection — using PyTorch. 
It includes training, evaluation, inference optimization (ONNX/TensorRT), Docker deployment, and MLflow experiment tracking.
Academic exercise only.

## 🔧 Tech Stack
- PyTorch
- ONNX, TensorRT
- MLflow
- Docker
- CI/CD ready

## 📂 Structure
- `src/`: Core training and model code
- `scripts/`: Utilities and CLI commands
- `mlflow_tracking/`: Experiment logging
- `notebooks/`: Exploratory notebooks
- `Dockerfile`: Reproducible container setup

## 🚀 Quickstart

```bash
# Build the container
docker build -t lane-detection .

# Run training
docker run -it lane-detection python src/train.py
