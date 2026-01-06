# Tech Challenge Fase 4 – Pós Graduação em Machine Learning


## Equipe responsável

- Beatriz do Nascimento Pessoa
- Murilo Rebequi Coelho
- Marianna Soares Veríssimo

## Requisitos do desafio

O desafio é criar um modelo preditivo de redes neurais Long Short Term Memory (LSTM) para predizer o valor de fechamento da bolsa de valores de uma empresa à escolha e realizar toda a pipeline de desenvolvimento, desde a criação do modelo preditivo até o deploy do modelo em uma API que permita a previsão de preços de ações.

## Estrutura do projeto

- `api/`: código da API FastAPI para servir previsões.
- `notebooks/`: notebooks do pipeline (EDA, janelas, treino LSTM, avaliação).
- `data/raw/`: dados brutos baixados do Yahoo Finance.
- `data/processed/`: dados limpos prontos para modelagem.
- `data/models/`: artefatos do modelo (pesos e scaler).

## Notebooks

- `01_eda.ipynb`: download/EDA dos dados e geração dos CSVs limpos.
- `02_windows.ipynb`: criação de janelas (lookback/horizon) e baselines.
- `03_lstm.ipynb`: treino da LSTM, curvas de perda e salvamento do modelo/scaler.
- `04_eval.ipynb`: avaliação no conjunto de teste e gráficos de comparação.




## API (FastAPI)

### Como rodar

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

### Como rodar com Docker

```bash
docker build -t tc4-api .
docker run -p 8000:8000 tc4-api
```

### Como rodar com Docker Compose

```bash
docker compose up --build
```

### Como rodar os testes

```bash
pip install -r requirements.txt
pytest
```

### Endpoints

- Health check: `GET /health`
- Predição: `POST /predict`

### Documentação

- `http://127.0.0.1:8000/docs`

### Exemplo de request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prices":[188.0695,188.1395,199.0389,193.1492,193.7892,186.8496,190.1594,186.5896,181.3499,186.5096,180.6299,178.8700,182.5398,177.8101,180.2500,176.9901,179.9100,181.4499,179.5800,183.3800,182.4100,185.5500,184.9700,183.7800,180.9300,175.0200,176.2900,177.7200,170.9400,174.1400]}'
```
