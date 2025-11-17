# Taglish Product Review Understanding with Multilingual Transformers

**Final Project - 3DEV Team**

## Repository

**Code Repository URL:** https://github.com/Bhawone/3dev-taglish-review-analyzer.git

**GitHub:** https://github.com/Bhawone/3dev-taglish-review-analyzer

---

## Project Overview

This project implements a comprehensive Taglish (Tagalog-English code-switched) sentiment analysis system for Philippine e-commerce reviews. The system includes:

- Overall sentiment classification (Negative, Neutral, Positive)
- Aspect-based sentiment analysis
- Deception detection
- Deployable API and UI

## Authors

- Bharon Christopher P. Candelaria
- Justin Clark A. Posadas
- Emjhay Theresa B. Tumulak

## Deliverables

### 1. Final IEEE Report
- **File:** `Final_Project_3DEV.docx`
- Comprehensive IEEE format report with methodology, results, and conclusions

### 2. Ground-Truth Test Set
- **File:** `ground_truth_test_set.csv`
- Contains 734 test samples with review and label columns
- Used for all final metric evaluations

### 3. Code Repository
- **URL:** https://github.com/Bhawone/3dev-taglish-review-analyzer
- All project code, models, and scripts

### 4. Experiment Logs and Checkpoints
- **Experiment Logs:** `exports/Experiments_All.xlsx`
- **Checkpoints:** `checkpoints/xlmrb/best/`
- Raw training logs and model checkpoints

### 5. Reproducibility Notebook
- **File:** `3DEV_Final_Project.ipynb`
- Automatically runs final evaluation on test set and generates key metrics
- All figures and evaluation results display inline in the notebook

---

## Setup Instructions

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Bhawone/3dev-taglish-review-analyzer.git
cd 3dev-taglish-review-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Preparation

1. Ensure `FiReCS.csv` is in the project root directory
2. The notebook will automatically create train/val/test splits
3. Or use the pre-generated split IDs:
   - `train_ids.csv`
   - `val_ids.csv`
   - `test_ids.csv`

### Running the Reproducibility Notebook

1. Open `3DEV_Final_Project.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells sequentially
3. The notebook will:
   - Load and preprocess data
   - Train baseline models
   - Fine-tune transformer models
   - Evaluate on test set
   - Generate metrics and visualizations (all displayed inline)

### Running the API

**Backend (FastAPI):**
```bash
uvicorn app.backend.main:app --reload
```

**Frontend (Streamlit):**
```bash
streamlit run app/ui/app.py
```

---

## File Structure

```
Elective 4_3DEFV/
├── 3DEV_Final_Project.ipynb      # Main reproducibility notebook
├── Final_Project_3DEV.docx        # IEEE report
├── ground_truth_test_set.csv     # Final test set
├── FiReCS.csv                     # Full dataset
├── train_ids.csv                  # Training set IDs
├── val_ids.csv                    # Validation set IDs
├── test_ids.csv                   # Test set IDs
├── checkpoints/                   # Model checkpoints
│   └── xlmrb/
│       └── best/                  # Best model checkpoint
├── exports/                       # Experiment logs and results
│   ├── Experiments_All.xlsx      # All experiment results
│   ├── Best_by_Model.xlsx         # Best results per model
│   └── ...
├── app/                           # Application code
│   ├── backend/                   # FastAPI backend
│   ├── inference/                 # Inference scripts
│   └── ui/                        # Streamlit UI
├── models/                        # Saved models
├── tools/                         # Utility scripts
└── requirements.txt               # Python dependencies
```

---

## Key Results

- **Test Accuracy:** ~81%
- **Test Macro-F1:** ~81%
- **Best Model:** XLM-RoBERTa-base fine-tuned on FiReCS

---

## Citation

If you use this work, please cite:

```
[TO BE ADDED - IEEE citation format]
```

---

## License

[TO BE ADDED]

---

## Contact

For questions or issues, please contact:
- Bharon Christopher P. Candelaria: bharonchristopher.candelaria@my.jru.edu
- Justin Clark A. Posadas: justinclark.posadas@my.jru.edu
- Emjhay Theresa B. Tumulak: emjhaytheresa.tumulak@my.jru.edu

