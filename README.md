 # Phishing Detector API

 This project is a FastAPI-based phishing email detection service powered by a Transformer model built in PyTorch (more info on the model [Transformer-based Phishing Detector](https://github.com/H1548/TransformerModel-PhishingDetector)). The API accepts email text/string as input, predicts whether the message is phishing-related, and returns structured advice such as recommended action, and user advice.

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
```
## Run Locally

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies
4. Start the FastAPI server

```bash
git clone <your-repo-url>
cd phishing-detector-api
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.app.main:app --reload
```
Open : http://localhost:8000/docs

## Run with Docker

Build the image:

```bash
docker build -t phishing-api .
```
Run the container 
```bash
docker run -p 8000:8000 phishing-api
```
Open : http://localhost:8000/docs

## Example Request

POST `/predict`

```json
{
  "input_data": "Subject: SecOps: malware infection on developer machine (security review) Good morning Jamie, Your developer machine triggered an automatic quarantine based on logs after detecting malware infection. Enable mfa right away: hxxps://revalidate-shieldops[.]invalid/secure?rid=530701 are infection. Best, Elliot"
}
```

## Example Response
```json
{
  "predictions": {
    "label": "phishing-credential",
    "recommended_action": "Do not click links or enter credentials; report and delete.",
    "user_advice": "Access accounts by typing the official URL or using a trusted app/bookmark. If you interacted, immediately change your password on the real site and revoke any suspicious sessions/MFA methods."
  }
}
```
## Model
The backend uses a Transformer-based phishing detection model developed in PyTorch. The model is loaded once at application startup and used during inference through the FastAPI `/predict` endpoint.

## Future Improvements
- Add batch inference support
- Add confidence scores
- Implement and improve logging and monitoring for production use

## Author

Hasan Farooq  
MSc Cybersecurity | Machine Learning | AI in Cybersecurity