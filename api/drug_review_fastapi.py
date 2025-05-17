from fastapi import FastAPI, HTTPException, Request, Header
from pydantic import BaseModel
import joblib
from dotenv import load_dotenv
import os

# Load model
app = FastAPI(title="Drug Review Sentiment API")
model = joblib.load("Models/Drug_reviews/best_model.pkl")

load_dotenv()
API_KEY = os.getenv("API_KEY")
class ReviewRequest(BaseModel):
    review: str

@app.post("/predict_drug_review")
def predict_sentiment(data: ReviewRequest, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    try:
        review = data.review
        prediction = model.predict([review])[0]
        
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        sentiment = label_map.get(prediction, "unknown")
        
        return {
            "review": data.review,
            "predicted_sentiment": sentiment
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))