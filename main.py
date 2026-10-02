from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from keras.models import load_model
import numpy as np
import pickle
import re


"""
1. Constants
A. Model Path (BiGRU)
B. Tokenizer Path
C. Max Sequence Length
D. Emotion Labels
E. Emotion Emojis
"""

# A. Model Path (BiGRU)
model_path = "Artifacts/BiGRU_Modle.keras"

# B. Tokenizer Path
tokenizer_path = "Artifacts/tokenizer.pkl"

# C. Max Sequence Length
max_sequence_length = 50

# D. Emotion Labels
emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# E. Emotion Emojis
EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}


"""
2. Preprocess the upcoming text
"""

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


"""
3. Request and Response Schemas
"""

class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={
            "example": "I feel so happy and excited"
        }
    )


class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilites: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


"""
4. Model Loading and LifeSpan Management
"""

dl_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the model and tokenizer...")

    dl_model["BiGRU"] = load_model(model_path)

    with open(tokenizer_path, "rb") as file:
        dl_model["Tokenizer"] = pickle.load(file)

    print("Model are loaded successfully...")

    yield

    dl_model.clear()


"""
5. FastAPI App
"""

app = FastAPI(
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


"""
6. API Endpoints
"""

# A. Server UI
@app.get("/", include_in_schema=False)
def server_ui():
    return FileResponse("static/index.html")


# B. Health Check
@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="Server is running",
        model_loaded=bool(dl_model)
    )


# C. Predict Emotion
@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):

    BiGRU_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("Tokenizer")

    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Please try again later."
        )

    # 1. Preprocess text
    cleaned_text = preprocess_text(text_input.text)

    # 2. Convert text into numbers
    tokenized_text = tokenizer_model.texts_to_sequences(
        [cleaned_text]
    )

    # 3. Pad sequence
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )

    # 4. Prediction
    probabilites = BiGRU_model.predict(
        padded_sequence
    )[0]

    top_emotion_index = int(
        np.argmax(probabilites)
    )

    # 5. All probabilities
    all_probabilites = {
        label: float(prob)
        for prob, label in zip(
            probabilites,
            emotion_labels
        )
    }

    # 6. Response
    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=emotion_labels[top_emotion_index],
        confidence=float(
            probabilites[top_emotion_index]
        ),
        all_probabilites=all_probabilites
    )