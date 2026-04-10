"""
train.py — Model Training Pipeline
Trains a specified model with hyperparameter tuning using GridSearchCV.

Usage:
    python src/train.py --data_path data/tv-shows.csv --model lr --seed 42
    python src/train.py --data_path data/tv-shows.csv --model svm --seed 42
    python src/train.py --data_path data/tv-shows.csv --model xgboost --seed 42
"""

import argparse
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


  
# MODEL CONFIGURATIONS
  

def get_model_and_grid(model_name, seed):
    """Return the model wrapped in OneVsRest and its hyperparameter grid."""

    if model_name == 'lr':
        model = OneVsRestClassifier(
            LogisticRegression(
                max_iter=1000,
                solver='liblinear',
                class_weight='balanced',
                random_state=seed
            ),
            n_jobs=-1
        )
        param_grid = {
            'estimator__C': [0.1, 0.3, 1.0, 3.0],
        }
        display_name = 'Logistic Regression'

    elif model_name == 'svm':
        model = OneVsRestClassifier(
            LinearSVC(
                max_iter=2000,
                class_weight='balanced',
                random_state=seed
            ),
            n_jobs=-1
        )
        param_grid = {
            'estimator__C': [0.1, 0.3, 1.0, 3.0],
        }
        display_name = 'Linear SVM'

    elif model_name == 'xgboost':
        from xgboost import XGBClassifier
        model = OneVsRestClassifier(
            XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                n_jobs=-1,
                random_state=seed,
                verbosity=0,
                tree_method='hist'
            ),
            n_jobs=1
        )
        param_grid = {
            'estimator__n_estimators': [100, 200],
            'estimator__max_depth': [4, 6],
            'estimator__learning_rate': [0.1, 0.2],
        }
        display_name = 'XGBoost'

    else:
        raise ValueError("Unknown model: " + model_name + ". Choose from: lr, svm, xgboost")

    return model, param_grid, display_name


  
# TRAINING FUNCTION
  

def train_model(X_train, y_train, model, param_grid, display_name):
    """Train a model with GridSearchCV and return the best estimator."""

    # Calculate grid size
    grid_size = 1
    for values in param_grid.values():
        grid_size = grid_size * len(values)

    print("Model: " + display_name)
    print("Grid: " + str(param_grid))
    print("Combinations: " + str(grid_size) + " x 3-fold CV = " + str(grid_size * 3) + " fits")
    print("")

    start = time.time()

    search = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring='f1_micro',
        cv=3,
        n_jobs=1,
        verbose=1
    )
    search.fit(X_train, y_train)

    train_time = time.time() - start

    print("")
    print("Done in " + str(round(train_time, 1)) + " seconds")
    print("Best params: " + str(search.best_params_))
    print("Best CV F1 Micro: " + str(round(search.best_score_, 4)))
    print("")

    # Show all CV results
    print("All combinations tried:")
    cv_df = pd.DataFrame(search.cv_results_)
    for _, row in cv_df.sort_values('rank_test_score').iterrows():
        score = round(row['mean_test_score'], 4)
        std = round(row['std_test_score'], 4)
        print("  " + str(row['params']) + "  ->  F1 = " + str(score) + " (+/-" + str(std) + ")")

    return search.best_estimator_, search.best_params_, train_time


  
# MAIN
  

def main():
    parser = argparse.ArgumentParser(description='Train a genre prediction model')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to preprocessed data pickle')
    parser.add_argument('--model', type=str, required=True, choices=['lr', 'svm', 'xgboost'],
                        help='Model to train: lr, svm, or xgboost')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='outputs/models',
                        help='Directory to save trained model')
    args = parser.parse_args()

    print("=" * 50)
    print("MODEL TRAINING")
    print("=" * 50)

    # Load preprocessed data
    print("Loading data from: " + args.data_path)
    with open(args.data_path, 'rb') as f:
        artifacts = pickle.load(f)

    X_train = artifacts['X_train']
    y_train = artifacts['y_train']
    print("Train set: " + str(X_train.shape[0]) + " samples, " + str(X_train.shape[1]) + " features")
    print("")

    # Get model and grid
    model, param_grid, display_name = get_model_and_grid(args.model, args.seed)

    # Train with GridSearchCV
    best_model, best_params, train_time = train_model(
        X_train, y_train, model, param_grid, display_name
    )

    # Save the trained model
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    model_filename = {
        'lr': 'logistic_regression.pkl',
        'svm': 'linear_svm.pkl',
        'xgboost': 'xgboost.pkl',
    }[args.model]

    output_path = os.path.join(args.output_dir, model_filename)
    with open(output_path, 'wb') as f:
        pickle.dump(best_model, f)

    print("")
    print("Model saved to: " + output_path)
    print("Done!")


if __name__ == '__main__':
    main()
