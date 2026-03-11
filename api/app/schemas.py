from pydantic import BaseModel, Field
from typing import List, Any



class PredictRequest(BaseModel):
    # customize according to your expected input format
    input_data: str = Field(min_length = 1, max_length = 800, description="The email content to classify")

class Predictions(BaseModel):
    label: str
    recommended_action: str
    user_advice: str

class PredictResponse(BaseModel):
    predictions: Predictions

