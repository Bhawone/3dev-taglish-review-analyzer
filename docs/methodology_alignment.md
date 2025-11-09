# Methodology Alignment Check

This note cross-checks the implemented pipeline in `EF3_AO_Candelaria_Notebook (1).ipynb`
against the proposed methodology (Sections IV–VI of the IEEE draft).

## Data & Preprocessing
- **Dataset**: Using the FiReCS corpus (`FiReCS.csv`) with persisted `train_ids.csv` /
  `val_ids.csv` / `test_ids.csv` aligns with the plan’s product-focused subset.
- **Cleaning & normalization**: The Notebook retains case, strips HTML/control characters,
  caps repeated characters, and records a code-switch ratio—matching Section V.B.
- **Splits**: Stratified 80/10/10 split implemented; reproducibility achieved via saved IDs.

## Baselines
- **TF-IDF + Logistic Regression**: Implemented with `baseline_predictions_test.csv`
  and logged to `runs_log.csv`, satisfying baseline requirements in Section VI.B.
- **Keyword heuristics for aspects/deception**: still pending; placeholders required when
  expanding to ABSA and deception modules.

## Transformer Fine-Tuning
- **Overall sentiment**: Sweep cell (`Cell 36`) instantiates XLM-RoBERTa with early stopping,
  class-weight toggle, and metric tracking. Needs a rerun to persist `transformer_predictions_test.csv`
  and populate `runs_log.csv` with transformer metrics.
- **Hyperparameter search**: Optuna block present (Cells 24–26) but disabled (`AUTO_TUNE_ENABLED=False`)
  after execution; metrics/export directories (`tuning/`) already populated.

## Aspect-Based Sentiment & Deception
- **Aspect span extraction**: Not yet implemented—Section V.C requires BIO tagging head
  and annotated spans (annotation work outstanding).
- **Per-aspect polarity**: Not yet implemented—requires either multi-head fine-tuning
  or post-processing pipeline.
- **Deception classifier**: Not yet implemented—Section V.C.4 still pending annotations
  and model definition.

## Evaluation & Reporting
- **Metrics consolidation**: `tools/summarize_metrics.py` now produces
  `exports/metrics_summary.(json|md)` with baseline results. Transformer metrics pending
  once predictions are available.
- **Artifacts**: Confusion matrices + comparison CSVs reside in `exports/` as planned.
- **Documentation**: Need to draft dataset/model cards, annotation guidelines,
  and integrate figures/tables into the IEEE manuscript.

## Action Items
1. Re-run transformer sweep to generate predictions & update `runs_log.csv`.
2. Finalize aspect-span annotations; implement BIO tagging trainer.
3. Add per-aspect polarity and deception heads (or stub heuristics) before prototype release.
4. Complete reporting artifacts (dataset/model cards, IEEE sections, dashboard screenshots).


