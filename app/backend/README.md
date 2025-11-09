## Taglish Review Analyzer API

Prototype FastAPI service that wraps the shared inference utilities in
`app/inference/predictor.py`. The API exposes:

- `GET /health` – basic readiness probe
- `POST /analyze` – accepts `{"review": "..."}` and returns sentiment scores,
  heuristic aspect highlights, and a deception score.

### Local development

1. Install dependencies (within your existing virtual environment):
   ```
   pip install fastapi uvicorn transformers scikit-learn pandas numpy
   ```
2. Start the API:
   ```
   uvicorn app.backend.main:app --reload
   ```
3. Open the interactive docs at http://127.0.0.1:8000/docs

> The service will automatically fall back to a TF-IDF baseline if the
> fine-tuned checkpoint is absent. Once the transformer sweep saves a model
> under `checkpoints/xlmrb/best`, it will be used for inference instead.


