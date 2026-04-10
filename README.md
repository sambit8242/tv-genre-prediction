# TV Show & Movie Genre Prediction

Multi-label genre prediction for TV shows and movies using traditional ML models (Logistic Regression, Linear SVM, XGBoost).

## Dataset

Netflix + Disney combined dataset with ~9,338 titles and 13 attributes.  
Source: https://github.com/vinayak-ensemble/Dataset-TV-Shows-OTT

## High Level Approach

1. **Data Understanding & EDA**: Explored all columns, found data quality issues (rating/duration swap, duplicate genre names from Netflix vs Disney sources)
2. **Preprocessing**: Normalized 84 duplicate genres to 41, lemmatized text descriptions, multi-hot encoded country (handles co-productions), split duration into movie_minutes + tv_seasons
3. **Modeling**: Trained 3 models using OneVsRestClassifier for multi-label classification, tuned with GridSearchCV (3-fold CV)
4. **Evaluation**: 9 metrics including F1 Micro/Macro and AUC-ROC Micro/Macro
5. **Explainability**: Linear model coefficients, XGBoost feature importance, per-prediction explanations

## Project Structure

```
tv-genre-prediction/
├── data/
│   └── tv-shows.csv                  # Raw dataset
├── src/
│   ├── preprocess.py                 # Data preprocessing pipeline
│   ├── train.py                      # Model training with GridSearchCV
│   └── evaluate.py                   # Model evaluation and plots
├── outputs/
│   ├── models/                       # Trained model pickles
│   ├── plots/                        # All generated plots
│   └── model_comparison.csv          # Results table
├── report/
│   └── Model_Evaluation_Report.docx  # Final report
├── requirements.txt
└── README.md
```

## Setup & Reproduce

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess the data

```bash
python src/preprocess.py --data_path data/tv-shows.csv --output_path outputs/preprocessed_data.pkl
```

### 3. Train models

```bash
python src/train.py --data_path outputs/preprocessed_data.pkl --model lr --seed 42
python src/train.py --data_path outputs/preprocessed_data.pkl --model svm --seed 42
python src/train.py --data_path outputs/preprocessed_data.pkl --model xgboost --seed 42
```

### 4. Evaluate all models and generate report plots

```bash
python src/evaluate.py --data_path outputs/preprocessed_data.pkl --models_dir outputs/models --plots_dir outputs/plots
```

This generates:
- `model_comparison.csv` — metrics table for all 3 models
- 6 plots for the report (comparison + explainability)

