Mlops-complte-on-Medical-Insurance-Cost-Dataset

End-to-end MLOps pipeline for predicting medical insurance charges using the Medical Insurance Cost dataset.
This repository demonstrates data versioning (DVC), experiment tracking (MLflow), CI/CD, containerization (Docker), model serving (FastAPI), and basic monitoring/infra artefacts.

Table of Contents

Project summary

Features

Repository structure

Getting started

Prerequisites

Install & setup

Data (DVC)

Run training & experiments (MLflow + DVC)

Run API (Docker / locally)

CI / CD

Tests

Useful commands

Files of interest

Contributing

License

Project summary

Predict insurance charges (regression) from features such as age, sex, bmi, children, smoker, and region. This repo provides a reproducible MLOps workflow with:

data ingestion & versioning (DVC)

preprocessing & featurization

model training and evaluation

experiment tracking with MLflow

model packaging and Dockerized serving (FastAPI)

CI / CD pipeline (GitHub Actions placeholder)

basic infra manifests (K8s/Helm placeholders)

Features

DVC-tracked raw and processed data

Parameterized pipeline (dvc.yaml / params.yaml)

MLflow experiment tracking and model registry integration

FastAPI endpoint for predictions (/predict)

Dockerfile to containerize the serving app

Unit tests for data, features and model sanity checks

Example notebooks for EDA

Repository structure
.
├── README.md
├── Dockerfile
├── app.py                      # FastAPI (or API wrapper)
├── dvc.yaml
├── dvc.lock
├── .dvcignore
├── requirements.txt
├── params.yaml
├── data/
│   └── raw/insurance.csv       # (DVC-tracked, not in git)
├── models/
│   └── model.pkl               # trained model (DVC or Git LFS)
├── src/
│   ├── data/
│   │   ├── make_dataset.py
│   │   └── validate.py
│   ├── features/
│   │   ├── featurize.py
│   │   └── scaler.pkl
│   ├── models/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── model.py
│   └── api/
│       └── app.py              # FastAPI app (if app.py at root is different)
├── notebooks/
│   └── note-pad.ipynb
├── reports/
│   └── evaluation/
├── templates/
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_model.py
└── .github/
    └── workflows/ci-cd.yml

Getting started
Prerequisites

Python 3.9+

git, docker (optional for container), dvc (v2+ recommended)

MLflow (for tracking)

(Optional) AWS S3 / GCS for DVC remote storage

Install & setup

Clone the repo:

git clone https://github.com/bharathmrr/Mlops-complte-on-Medical-Insurance-Cost-Dataset.git
cd Mlops-complte-on-Medical-Insurance-Cost-Dataset


Create and activate virtualenv:

python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\activate       # Windows


Install dependencies:

pip install -r requirements.txt

Data (DVC)

Add a remote (example S3) and fetch data (if remote configured):

# (only if a DVC remote is configured for the repo)
dvc pull


If you have the dataset locally, place it:

data/raw/insurance.csv


and add to DVC (if you need to version it):

dvc add data/raw/insurance.csv
git add data/.gitignore data/raw/insurance.csv.dvc
git commit -m "Add raw insurance dataset to DVC"
dvc remote add -d storage s3://your-dvc-bucket/path
dvc push

Run training & experiments (MLflow + DVC)

Reproduce the DVC pipeline:

dvc repro


Train manually (example):

python src/models/train.py --data data/processed/insurance_processed.csv --output models/model.pkl


Log experiments with MLflow (example):

# start mlflow server (local)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns -p 5000

# from training script ensure mlflow.set_tracking_uri("http://localhost:5000")

Run API (Docker / locally)

Locally
If app.py (or src/api/app.py) exists:

uvicorn app:app --host 0.0.0.0 --port 8080
# or
uvicorn src.api.app:app --host 0.0.0.0 --port 8080


Request:

curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 29, "sex": "female", "bmi": 26.2, "children": 1, "smoker": "no", "region": "northwest"}'


Docker
Build image:

docker build -t insurance-api:latest .


Run container (pass model path or mounted volume):

docker run -p 8080:8080 -e MODEL_PATH=/app/models/model.pkl insurance-api:latest
# OR mount local models
docker run -p 8080:8080 -v $(pwd)/models:/app/models insurance-api:latest

CI / CD

.github/workflows/ci-cd.yml contains CI steps (lint, tests, DVC check, build docker).

Typical pipeline stages:

run unit tests (pytest)

DVC metadata check (dvc status)

build & push docker image (on release or main)

(optional) deploy to staging k8s cluster

Tests

Run tests:

pytest -q


Test examples include:

tests/test_data.py — schema & basic validation

tests/test_features.py — featurization consistency checks

tests/test_model.py — model prediction shape & basic sanity

Useful commands
# DVC
dvc status
dvc repro
dvc metrics show

# MLflow
mlflow ui   # start local UI (default port 5000)

# Docker
docker build -t insurance-api:latest .
docker run -p 8080:8080 insurance-api:latest

# Run tests
pytest -q

Files of interest

dvc.yaml / dvc.lock — DVC pipeline stages

params.yaml — parameters used by pipeline & training

src/models/train.py — training script (logs to MLflow)

src/features/featurize.py — feature engineering pipeline

src/api/app.py or app.py — FastAPI serving code

Dockerfile — containerization

.github/workflows/ci-cd.yml — CI pipeline

How to extend

Add model registry integration for automatic deployment on MLflow model registration.

Add Prometheus exporters and Grafana dashboards for monitoring predictions & latency.

Add automated data drift detection (e.g., Evidently.ai or custom drift tests).

Add K8s manifests + Helm charts for production deployment.

Contributing

Contributions welcome — open an issue or PR. Suggested workflow:

Fork repo

Create feature branch

Add tests for new functionality

Open a PR and reference related issue

Notes

Keep sensitive credentials out of the repo — use environment variables or secret stores for DVC remotes, S3/GCS credentials, and MLflow backend storage URIs.

Ensure data/raw/insurance.csv is DVC-tracked and not committed to Git.

License

This project is licensed under the MIT License — see LIC
