from pathlib import Path
from typing import List

import joblib
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("data/models/lstm_nvda.pt")
SCALER_PATH = Path("data/models/scaler_nvda.pkl")

LOOKBACK = 30 # 30 preços
HORIZON = 1

app = FastAPI(title="LSTM Price Prediction API")


class LSTMModel(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class PredictRequest(BaseModel):
    prices: List[float] = Field(..., min_items=LOOKBACK, max_items=LOOKBACK)


class PredictResponse(BaseModel):
    prediction: float


device = torch.device("cpu")
model: LSTMModel | None = None
scaler = None


@app.on_event("startup")
def startup():
    global model, scaler

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modelo não encontrado em {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise RuntimeError(f"Scaler não encontrado em {SCALER_PATH}")

    model = LSTMModel().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    scaler = joblib.load(SCALER_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado")

    if len(req.prices) != LOOKBACK:
        raise HTTPException(status_code=422, detail=f"prices deve ter tamanho {LOOKBACK}")

    prices = np.array(req.prices, dtype=np.float32).reshape(-1, 1)
    prices_scaled = scaler.transform(prices)

    X = prices_scaled.reshape(1, LOOKBACK, 1)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    with torch.inference_mode():
        pred_scaled = model(X_t).cpu().numpy().reshape(-1, 1)
        pred = scaler.inverse_transform(pred_scaled).ravel()[0]

    return PredictResponse(prediction=float(pred))
