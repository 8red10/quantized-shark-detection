# Quantized Shark Detection
Explores the accuracy-latency-power Pareto frontier derived from quantizing object detection models of varying architectures and deploying them on the edge. 

All experiment stages can be found in this repository.

## 1. Data Preparation
Creates the train/val/test splits while ensuring near-duplicate images are kept to a single split to help prevent memorization. Also, identifies the calibration set for INT8 quantization.

Hardware = local compute

## 2. Training
Trains each model using frameworks and pipelines tuned to that model architecture. After training, exports each model to ONNX for compatibility with quantization. 

Hardware = cloud GPU

## 3. Edge Deployment
Quantizes and benchmarks each model to record accuracy, latency and power when deployed on the edge.

Hardware = Jetson Orin Nano

# Project Tree
```
my-project/
├── pyproject.toml
├── uv.lock
├── README.md
│
├── data/
├── manifests/
├── configs/
├── models/
│
└── packages/
    ├── common/
    │   ├── pyproject.toml
    │   └── src/
    │       └── common/
    │           ├── __init__.py
    │           ├── io.py
    │           ├── config.py
    │           └── utils.py
    │
    ├── data_prep/
    │   ├── pyproject.toml
    │   └── src/
    │       └── data_prep/
    │           ├── __init__.py
    │           └── __main__.py
    │
    ├── training/
    │   ├── pyproject.toml
    │   └── src/
    │       └── training/
    │           ├── __init__.py
    │           └── __main__.py
    │
    └── edge/
        ├── pyproject.toml
        └── src/
            └── edge/
                ├── __init__.py
                └── __main__.py
```