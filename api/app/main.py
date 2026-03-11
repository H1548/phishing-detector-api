from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import torch 
from phishing_model.FineTuningmodel import __version__ as model_version
from phishing_model.FineTuningmodel import Transformer
from phishing_model.Prompter import prompt 
from phishing_model.paths import Params_Path
from .schemas import PredictRequest, PredictResponse



def load_model():
    num_layers = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    CUDA_LAUNCH_BLOCKING=1
    loading_model = Transformer(num_layers)
    loading_model = loading_model.to(device)
    checkpoint = torch.load(Params_Path,  map_location=torch.device('cpu'))
    loading_model.load_state_dict(checkpoint['model_state_dict'])
    loading_model.eval()
    return loading_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()
    yield
    

app = FastAPI(title="Phishing Detector API", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok", "model_version": model_version}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
        
    input_str = request.input_data.strip()

    if not input_str:
        raise HTTPException(status_code=400, detail="input_data cannot be empty or whitespace only")

    output = prompt(input_str, model)

    return PredictResponse(predictions = output)
