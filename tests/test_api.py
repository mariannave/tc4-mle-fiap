import torch
from fastapi.testclient import TestClient

from api import main


class DummyScaler:
    def transform(self, values):
        return values

    def inverse_transform(self, values):
        return values


class DummyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[2.0]], device=x.device)


def create_client():
    main.app.router.on_startup = []
    return TestClient(main.app)


def test_health():
    client = create_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_success():
    client = create_client()
    main.model = DummyModel()
    main.scaler = DummyScaler()

    payload = {"prices": [1.0] * main.LOOKBACK}
    resp = client.post("/predict", json=payload)

    assert resp.status_code == 200
    assert resp.json()["prediction"] == 2.0


def test_predict_invalid_length():
    client = create_client()
    main.model = DummyModel()
    main.scaler = DummyScaler()

    payload = {"prices": [1.0] * (main.LOOKBACK - 1)}
    resp = client.post("/predict", json=payload)

    assert resp.status_code == 422


def test_predict_model_not_loaded():
    client = create_client()
    main.model = None
    main.scaler = None

    payload = {"prices": [1.0] * main.LOOKBACK}
    resp = client.post("/predict", json=payload)

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Modelo não carregado"
