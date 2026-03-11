 # Phishing Detector API

 This project is a FastAPI-based phishing email detection service powered by a Transformer model built in PyTorch (more info on the model [text](https://github.com/H1548/TransformerModel-PhishingDetector)). The API accepts email text/string as input, predicts whether the message is phishing-related, and returns structured advice such as recommended action, and user advice.

 ## Motivation

 Phishing emails remain one of the most common cyber attack vectors. This project explores how machine learning can be used to automate phishing detection and provide actionable response guidance through an API service.

 ## Features

- FastAPI endpoint for phishing prediction
- Transformer-based PyTorch model
- Structured JSON response with prediction and advice
- Dockerized deployment
- Health check endpoint
- Input validation and error handling

## Tech Stack
- Python
- FastAPI
- PyTorch
- Pydantic
- Uvicorn
- Docker

## Project Structure

```text
Phishing-Detector-api/
├── api/
│   └── app/
│       └── main.py
|       └── schemas.py
├── phishing_model/
|   ├── __init__.py
│   ├── FineTuningmodel.py
│   ├── paths.py
│   └── Prompter.py
|   └── utils.py
├── .dockerignore
├── requirements.txt
├── Dockerfile
├── LICENSE
├── requirements.txt
└── README.md