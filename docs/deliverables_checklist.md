# Deliverables Checklist

Use this list to prepare the submission bundle and presentation package.

## Data & Annotation Artefacts
- [ ] Confirm FiReCS license notice, document any additional product-specific samples.
- [ ] Finalize aspect-span and deception annotation guidelines (PDF/Markdown).
- [ ] Release cleaned splits (`train_ids.csv`, `val_ids.csv`, `test_ids.csv`) with README.

## Models & Code
- [ ] Persist best transformer checkpoint under `checkpoints/xlmrb/best`.
- [ ] Export ONNX/quantized variants (when dependencies available).
- [ ] Provide training/evaluation scripts (`tools/summarize_metrics.py`, notebooks).

## Inference & Prototype
- [ ] FastAPI backend (`app/backend/main.py`) with quickstart instructions.
- [ ] Streamlit UI (`app/ui/app.py`) configured with sample prompts.
- [ ] Example request/response pairs saved in `results/demo_samples.json`.

## Reporting
- [ ] Update `exports/metrics_summary.*` after transformer evaluation.
- [ ] Insert confusion matrices and comparison tables into IEEE manuscript.
- [ ] Draft dataset & model cards, cite ethical considerations.

## Deployment & Ops
- [ ] Dockerfile / requirements.txt capturing runtime dependencies.
- [ ] API documentation (OpenAPI already available via FastAPI docs).
- [ ] Testing checklist covering unit, smoke, and manual UI tests.


