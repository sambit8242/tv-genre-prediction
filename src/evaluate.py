"""
evaluate.py — Model Evaluation & Explainability Pipeline
Loads all trained models, evaluates them, generates comparison plots
and explainability visuals for the report.

Usage:
    python src/evaluate.py --data_path outputs/preprocessed_data.pkl --models_dir outputs/models --plots_dir outputs/plots
"""

import argparse
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.multiclass import _ConstantPredictor


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def compute_metrics(model_name, y_true, y_pred, y_score):
    """Compute all 9 evaluation metrics for a model."""
    results = {
        'Model': model_name,
        'F1 Micro': round(f1_score(y_true, y_pred, average='micro', zero_division=0), 4),
        'F1 Macro': round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'F1 Weighted': round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'Precision Micro': round(precision_score(y_true, y_pred, average='micro', zero_division=0), 4),
        'Recall Micro': round(recall_score(y_true, y_pred, average='micro', zero_division=0), 4),
    }

    try:
        valid = (y_true.sum(axis=0) > 0) & (y_true.sum(axis=0) < y_true.shape[0])
        results['AUC Micro'] = round(roc_auc_score(y_true[:, valid], y_score[:, valid], average='micro'), 4)
        results['AUC Macro'] = round(roc_auc_score(y_true[:, valid], y_score[:, valid], average='macro'), 4)
    except:
        results['AUC Micro'] = None
        results['AUC Macro'] = None

    return results


def get_scores(model, X_test):
    """Get probability scores or decision function values for AUC."""
    if hasattr(model, 'predict_proba'):
        try:
            return model.predict_proba(X_test)
        except:
            pass
    if hasattr(model, 'decision_function'):
        return model.decision_function(X_test)
    return model.predict(X_test).astype(float)


def get_coefs(ovr_model, n_genres, n_features):
    """Extract coefficient matrix from a linear OneVsRest model."""
    coefs = np.zeros((n_genres, n_features))
    for i, est in enumerate(ovr_model.estimators_):
        if isinstance(est, _ConstantPredictor):
            continue
        elif hasattr(est, 'coef_'):
            coef = est.coef_
            if hasattr(coef, 'toarray'):
                coef = coef.toarray()
            coefs[i] = coef.flatten()
    return coefs


def per_genre_auc(y_true, y_score):
    """Compute AUC for each genre separately."""
    aucs = []
    for i in range(y_true.shape[1]):
        if 0 < y_true[:, i].sum() < y_true.shape[0]:
            try:
                aucs.append(roc_auc_score(y_true[:, i], y_score[:, i]))
            except:
                aucs.append(np.nan)
        else:
            aucs.append(np.nan)
    return np.array(aucs)


# ============================================================
# PLOT FUNCTIONS
# ============================================================

def plot_model_comparison(all_results, plots_dir):
    """Plot F1 and AUC comparison for all tuned models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = list(all_results.index)
    colors = ['blue', 'red', 'purple']

    # Left: F1 metrics
    ax = axes[0]
    metrics = ['F1 Micro', 'F1 Macro', 'F1 Weighted']
    x = np.arange(len(metrics))
    width = 0.25

    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [all_results.loc[model, m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=model, color=color)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison - F1 & Accuracy')
    ax.legend(fontsize=9)

    # Right: AUC metrics
    ax = axes[1]
    auc_metrics = ['AUC Micro', 'AUC Macro']
    x = np.arange(len(auc_metrics))

    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [all_results.loc[model, m] for m in auc_metrics]
        ax.bar(x + i * width, vals, width, label=model, color=color)

    ax.set_xticks(x + width)
    ax.set_xticklabels(auc_metrics)
    ax.set_ylabel('AUC Score')
    ax.set_title('Model Comparison - AUC-ROC')
    ax.legend(fontsize=9)
    ax.set_ylim(0.5, 1.0)

    plt.tight_layout()
    path = os.path.join(plots_dir, '17_tuned_model_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: " + path)


def plot_per_genre_f1(y_test, predictions, genre_names, plots_dir):
    """Plot per-genre F1 for all models."""
    top20_idx = np.argsort(y_test.sum(axis=0))[::-1][:20]
    top20_names = [genre_names[i] for i in top20_idx]

    model_names = list(predictions.keys())
    colors = ['blue', 'red', 'purple']
    x = np.arange(len(top20_idx))
    width = 0.27

    plt.figure(figsize=(14, 7))
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        y_pred = predictions[model_name]
        f1_per = f1_score(y_test, y_pred, average=None, zero_division=0)
        offset = (i - 1) * width
        plt.bar(x + offset, [f1_per[j] for j in top20_idx], width, label=model_name, color=color)

    plt.xticks(x, top20_names, rotation=45, ha='right')
    plt.ylabel('F1 Score')
    plt.title('F1 Score per Genre - Top 20 Genres')
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(plots_dir, '18_per_genre_f1.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: " + path)


def plot_per_genre_auc(y_test, scores, genre_names, plots_dir):
    """Plot per-genre AUC for all models."""
    top20_idx = np.argsort(y_test.sum(axis=0))[::-1][:20]
    top20_names = [genre_names[i] for i in top20_idx]

    model_names = list(scores.keys())
    colors = ['blue', 'red', 'purple']
    x = np.arange(len(top20_idx))
    width = 0.27

    plt.figure(figsize=(14, 7))
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        y_score = scores[model_name]
        aucs = per_genre_auc(y_test, y_score)
        offset = (i - 1) * width
        plt.bar(x + offset, [aucs[j] for j in top20_idx], width, label=model_name, color=color)

    plt.xticks(x, top20_names, rotation=45, ha='right')
    plt.ylabel('AUC-ROC')
    plt.title('AUC-ROC per Genre - Top 20 Genres')
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    plt.ylim(0.4, 1.05)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(plots_dir, '19_per_genre_auc.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: " + path)


def plot_lr_top_features(lr_coefs, feature_names, genre_names, plots_dir):
    """Plot top 10 positive features for 6 genres using LR coefficients."""
    genres_to_show = ['Horror', 'Family', 'Documentary', 'Romance', 'Stand-Up Comedy', 'Action & Adventure']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Top 10 Positive Features by Genre (Logistic Regression)', fontsize=14)

    for idx, genre_name in enumerate(genres_to_show):
        ax = axes[idx // 3][idx % 3]

        if genre_name not in genre_names:
            ax.set_title(genre_name + ' (not found)')
            continue

        genre_idx = genre_names.index(genre_name)
        weights = lr_coefs[genre_idx]
        top10_idx = np.argsort(weights)[-10:]

        names = [feature_names[i] for i in top10_idx]
        values = [weights[i] for i in top10_idx]

        ax.barh(range(len(names)), values)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_title(genre_name)

    plt.tight_layout()
    path = os.path.join(plots_dir, '20_lr_top_features.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: " + path)




def plot_svm_top_features(svm_coefs, feature_names, genre_names, plots_dir):
    """Plot top 10 positive features for 6 genres using SVM coefficients."""
    genres_to_show = ['Horror', 'Family', 'Documentary', 'Romance', 'Stand-Up Comedy', 'Action & Adventure']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Top 10 Positive Features by Genre (Linear SVM - Recommended Model)', fontsize=14)

    for idx, genre_name in enumerate(genres_to_show):
        ax = axes[idx // 3][idx % 3]

        if genre_name not in genre_names:
            ax.set_title(genre_name + ' (not found)')
            continue

        genre_idx = genre_names.index(genre_name)
        weights = svm_coefs[genre_idx]
        top10_idx = np.argsort(weights)[-10:]

        names = [feature_names[i] for i in top10_idx]
        values = [weights[i] for i in top10_idx]

        ax.barh(range(len(names)), values, color='red')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_title(genre_name)

    plt.tight_layout()
    path = os.path.join(plots_dir, '20b_svm_top_features.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: " + path)

def plot_xgb_feature_importance(xgb_model, n_features, feature_names, genre_names, plots_dir):
    """Plot top 20 globally important features from XGBoost."""
    xgb_importances = []

    for est in xgb_model.estimators_:
        imp_array = np.zeros(n_features)
        if hasattr(est, 'get_booster'):
            booster = est.get_booster()
            score_dict = booster.get_score(importance_type='gain')
            for feat_key, score in score_dict.items():
                feat_idx = int(feat_key[1:])
                imp_array[feat_idx] = score
        xgb_importances.append(imp_array)

    xgb_importances = np.array(xgb_importances)
    global_xgb = xgb_importances.mean(axis=0)

    top20_idx = np.argsort(global_xgb)[::-1][:20]
    top20_names = [feature_names[i] for i in top20_idx][::-1]
    top20_values = global_xgb[top20_idx][::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top20_names)), top20_values, color='purple')
    plt.yticks(range(len(top20_names)), top20_names, fontsize=10)
    plt.xlabel('Average Feature Importance (Gain)')
    plt.title('Top 20 Globally Important Features (XGBoost)')
    plt.tight_layout()

    path = os.path.join(plots_dir, '21_xgb_feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: " + path)

    return global_xgb


def plot_cross_model_importance(lr_coefs, svm_coefs, global_xgb, feature_names, plots_dir):
    """Compare feature importance across all 3 models."""
    lr_importance = np.abs(lr_coefs).mean(axis=0)
    svm_importance = np.abs(svm_coefs).mean(axis=0)

    def normalize(x):
        if x.max() - x.min() == 0:
            return x
        return (x - x.min()) / (x.max() - x.min())

    lr_norm = normalize(lr_importance)
    svm_norm = normalize(svm_importance)
    xgb_norm = normalize(global_xgb)

    # Union of top 10 from each model
    top10_lr = set(np.argsort(lr_norm)[::-1][:10])
    top10_svm = set(np.argsort(svm_norm)[::-1][:10])
    top10_xgb = set(np.argsort(xgb_norm)[::-1][:10])
    union = sorted(list(top10_lr | top10_svm | top10_xgb),
                   key=lambda i: -(lr_norm[i] + svm_norm[i] + xgb_norm[i]))[:20]

    union_names = [feature_names[i] for i in union][::-1]
    lr_vals = [lr_norm[i] for i in union][::-1]
    svm_vals = [svm_norm[i] for i in union][::-1]
    xgb_vals = [xgb_norm[i] for i in union][::-1]

    y = np.arange(len(union_names))
    height = 0.27

    plt.figure(figsize=(12, 10))
    plt.barh(y - height, lr_vals, height, label='Logistic Regression', color='blue')
    plt.barh(y, svm_vals, height, label='Linear SVM', color='red')
    plt.barh(y + height, xgb_vals, height, label='XGBoost', color='purple')
    plt.yticks(y, union_names, fontsize=10)
    plt.xlabel('Normalized Importance (0-1)')
    plt.title('Feature Importance: All 3 Models Compared')
    plt.legend()
    plt.tight_layout()

    path = os.path.join(plots_dir, '22_cross_model_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: " + path)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate all models and generate report plots')
    parser.add_argument('--data_path', type=str, default='outputs/preprocessed_data.pkl',
                        help='Path to preprocessed data pickle')
    parser.add_argument('--models_dir', type=str, default='outputs/models',
                        help='Directory containing trained model pickles')
    parser.add_argument('--plots_dir', type=str, default='outputs/plots',
                        help='Directory to save plots')
    args = parser.parse_args()

    print("=" * 50)
    print("MODEL EVALUATION & EXPLAINABILITY")
    print("=" * 50)

    # Load data
    print("Loading data from: " + args.data_path)
    with open(args.data_path, 'rb') as f:
        artifacts = pickle.load(f)

    X_test = artifacts['X_test']
    y_test = artifacts['y_test']
    genre_names = artifacts['genre_names']
    feature_names = artifacts['feature_names']
    n_features = len(feature_names)
    print("Test set: " + str(X_test.shape[0]) + " samples, " + str(n_features) + " features")
    print("Genres: " + str(len(genre_names)))

    # Load all 3 models
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Linear SVM': 'linear_svm.pkl',
        'XGBoost': 'xgboost.pkl',
    }

    models = {}
    for name, filename in model_files.items():
        path = os.path.join(args.models_dir, filename)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
            print("Loaded: " + name)
        else:
            print("WARNING: " + path + " not found, skipping " + name)

    os.makedirs(args.plots_dir, exist_ok=True)

    # ========================================
    # STEP 1: Compute metrics for all models
    # ========================================
    print("")
    print("COMPUTING METRICS")
    print("-" * 50)

    all_results = []
    predictions = {}
    scores = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_score = get_scores(model, X_test)

        predictions[name] = y_pred
        scores[name] = y_score

        result = compute_metrics(name, y_test, y_pred, y_score)
        all_results.append(result)

        print("")
        print(name + ":")
        for key, value in result.items():
            if key != 'Model':
                print("  " + str(key).ljust(20) + ": " + str(value))

    results_df = pd.DataFrame(all_results).set_index('Model')

    # Save metrics to CSV
    csv_path = os.path.join(os.path.dirname(args.plots_dir), 'model_comparison.csv')
    results_df.to_csv(csv_path)
    print("")
    print("Saved metrics: " + csv_path)

    # ========================================
    # STEP 2: Generate comparison plots
    # ========================================
    print("")
    print("GENERATING COMPARISON PLOTS")
    print("-" * 50)

    # Plot 1: Model comparison (F1 + AUC)
    plot_model_comparison(results_df, args.plots_dir)

    # Plot 2: Per-genre F1
    plot_per_genre_f1(y_test, predictions, genre_names, args.plots_dir)

    # Plot 3: Per-genre AUC
    plot_per_genre_auc(y_test, scores, genre_names, args.plots_dir)

    # ========================================
    # STEP 3: Generate explainability plots
    # ========================================
    print("")
    print("GENERATING EXPLAINABILITY PLOTS")
    print("-" * 50)

    # Plot 4: LR top features per genre
    if 'Logistic Regression' in models:
        lr_coefs = get_coefs(models['Logistic Regression'], len(genre_names), n_features)
        plot_lr_top_features(lr_coefs, feature_names, genre_names, args.plots_dir)
    else:
        lr_coefs = None
        print("  Skipping LR features (model not found)")

    # Plot 5: SVM top features per genre (our recommended model)
    if 'Linear SVM' in models:
        svm_coefs = get_coefs(models['Linear SVM'], len(genre_names), n_features)
        plot_svm_top_features(svm_coefs, feature_names, genre_names, args.plots_dir)
    else:
        svm_coefs = None
        print("  Skipping SVM features (model not found)")

    # Plot 6: XGBoost feature importance
    if 'XGBoost' in models:
        global_xgb = plot_xgb_feature_importance(
            models['XGBoost'], n_features, feature_names, genre_names, args.plots_dir
        )
    else:
        global_xgb = None
        print("  Skipping XGBoost importance (model not found)")

    # Plot 7: Cross-model comparison (all 3)
    if lr_coefs is not None and svm_coefs is not None and global_xgb is not None:
        plot_cross_model_importance(lr_coefs, svm_coefs, global_xgb, feature_names, args.plots_dir)
    else:
        print("  Skipping cross-model plot (need all 3 models)")

    # ========================================
    # SUMMARY
    # ========================================
    print("")
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)

    best_model = results_df['F1 Micro'].idxmax()
    best_score = results_df['F1 Micro'].max()
    print("Best model (F1 Micro): " + best_model + " (" + str(best_score) + ")")
    print("")
    print("Plots generated:")
    print("  17_tuned_model_comparison.png  — F1 & AUC bars")
    print("  18_per_genre_f1.png            — F1 per genre (all models)")
    print("  19_per_genre_auc.png           — AUC per genre (all models)")
    print("  20_lr_top_features.png         — Top words per genre (LR)")
    print("  20b_svm_top_features.png       — Top words per genre (SVM - recommended)")
    print("  21_xgb_feature_importance.png  — XGBoost global importance")
    print("  22_cross_model_importance.png  — LR vs XGBoost comparison")
    print("")
    print("Done!")


if __name__ == '__main__':
    main()