# ResumeIQ — Explainable AI (XAI) Resume Screening System

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-EB6B02?style=flat)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-0052FF?style=flat)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ResumeIQ is an easy-to-use resume screening tool. It reads resume files (`.pdf` and `.docx`), extracts details like skills and experience, filters candidates using simple rules, and uses machine learning to score them. Most importantly, it shows you exactly **why** a candidate was shortlisted or rejected using explainable AI charts.

---

## Table of Contents

- [What the Project Does](#what-the-project-does)
- [Why the Project Is Useful](#why-the-project-is-useful)
- [Architecture & Tech Stack](#architecture--tech-stack)
  - [Repository Structure](#repository-structure)
  - [Model Training & Dataset](#model-training--dataset)
- [How to Run the Project](#how-to-run-the-project)
  - [Prerequisites](#prerequisites)
  - [Step 1: Install Requirements](#step-1-install-requirements)
  - [Step 2: Run the Backend API](#step-2-run-the-backend-api)
  - [Step 3: Open the Web Dashboard](#step-3-open-the-web-dashboard)
- [Usage Examples](#usage-examples)
  - [API Endpoints](#api-endpoints)
  - [cURL Request Example](#curl-request-example)
  - [Python Request Example](#python-request-example)
- [Where to Get Help](#where-to-get-help)

---

## What the Project Does

Here is how ResumeIQ works step-by-step:

1. **Extracts Text**: Reads text from PDF or Word (`.docx`) resume files.
2. **Parses Resume Details**: Uses Natural Language Processing (NLP) to find information like CGPA, skills, certifications, internships, and work experience.
3. **Checks Hard Rules**: Checks if the candidate meets minimum requirements (like CGPA >= 6.5 or skills score >= 6) and filters them out if they do not.
4. **Predicts Match Score**: Feeds the details into an XGBoost AI model to predict how well the candidate matches.
5. **Explains the Score**: Uses SHAP to show which factors (like strong skills or lack of experience) helped or hurt the score.
6. **Web Dashboard**: Provides a simple dark-themed website where you can drag-and-drop resumes and view results instantly.

---

## Why the Project Is Useful

* **No Black Box**: You see exactly why the AI made its decision (e.g. strong academic background, lacking internships).
* **Custom Rules**: You can easily change the minimum requirements (like minimum CGPA) in the [config.json](config.json) file.
* **Smart Text Reading**: Uses advanced word processing so it does not just match keywords blindly, but understands variation in words (e.g., matching "certified" and "certifications").
* **Friendly Dashboard**: Simple, interactive dashboard that runs locally on your machine.

---

## Architecture & Tech Stack

The system has two main parts:
* **Backend**: FastAPI (Python web server), XGBoost (AI model), NLTK (text parser), and SHAP (explanation tool).
* **Frontend**: A simple webpage ([index.html](index.html)) using HTML, CSS, and Chart.js.

### Repository Structure

* [main.py](main.py) - The FastAPI server code.
* [parser.py](parser.py) - The script that reads and extracts details from resumes.
* [config.json](config.json) - Configuration file containing active rules and threshold scores.
* [index.html](index.html) - The webpage frontend dashboard.
* [train.ipynb](train.ipynb) - The Jupyter notebook used to train the machine learning model.
* [requirements.txt](requirements.txt) - List of Python packages to install.
* [model.pkl](model.pkl) & [scaler.pkl](scaler.pkl) - The trained AI model and scaler files.
* [features.pkl](features.pkl) - List of feature names for the model.

### Model Training & Dataset

The machine learning model was trained and evaluated using:
* **Training Dataset**: [Resume Screening Dataset (200k Candidates)](https://www.kaggle.com/datasets/rhythmghai/resume-screening-dataset-200k-candidates) available on Kaggle.
* **Pipeline Notebook**: The training workflow, feature scaling, model training, and SHAP explainer configurations are implemented in [train.ipynb](train.ipynb).

---

## How to Run the Project

### Prerequisites

Make sure you have Python 3.8 or higher installed on your computer.

### Step 1: Install Requirements

Open your terminal in the project folder and run:
```bash
pip install -r requirements.txt
```
> [!NOTE]
> The first time you run the parser, it will automatically download necessary language files from NLTK.

### Step 2: Run the Backend API

Start the backend server by running:
```bash
python -m uvicorn main:app --reload
```
The API will now be running at `http://127.0.0.1:8000`. You can see the interactive API documentation at `http://127.0.0.1:8000/docs`.

### Step 3: Open the Web Dashboard

1. Locate the [index.html](index.html) file.
2. Double-click it to open it directly in your web browser (or serve it with a local server like Node's `npx serve` or Python's `python -m http.server 8080`).
3. Make sure the indicator in the top right says **API ONLINE** (green).

---

## Usage Examples

### API Endpoints

The FastAPI server exposes the following endpoints:

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/` | Welcoming message & available API endpoints. |
| `GET` | `/health` | Health check endpoint returning backend status and current classification threshold. |
| `GET` | `/rules` | Fetches the active hard screening rules loaded from `config.json`. |
| `POST` | `/predict` | Accepts a multipart file upload (`.pdf` or `.docx` resume) and returns the decision, probability, and SHAP explanation. |

### cURL Request Example

You can test the prediction endpoint directly from your terminal:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/resume.pdf"
```

### Python Request Example

Here is how you can programmatically query the screening API using Python:

```python
import requests

url = "http://127.0.0.1:8000/predict"
file_path = "path/to/candidate_cv.docx"

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    response = requests.post(url, files=files)

if response.status_code == 200:
    result = response.json()
    print(f"Status: {result['status']}")
    print(f"Match Probability: {result['probability']}%")
    print(f"Passed Hard Rules: {result['passed_hard_rules']}")
    
    if result['status'] == "SHORTLISTED":
        print("\nTop Positive Factors Driving Selection:")
        for driver in result['explanation']['positive_drivers'][:2]:
            print(f"- {driver['feature']}: Impact value of {driver['impact']}")
else:
    print(f"Failed with code {response.status_code}: {response.text}")
```

---

## Where to Get Help

If you run into issues, check these options:

* **Interactive Docs**: Go to `http://127.0.0.1:8000/docs` in your browser to test the API directly.
* **Server Logs**: Check your terminal console where FastAPI is running for error messages or warnings.
* **CORS Settings**: If the dashboard has trouble connecting to the backend, make sure CORS is allowed in [main.py](main.py) (it is set to `["*"]` by default).
