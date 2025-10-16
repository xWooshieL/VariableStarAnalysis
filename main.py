"""
Main script for running variable star analysis experiments.
Demonstrates usage of all modules with UCR StarLightCurves dataset.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our modules
from lib.data_loader import load_ucr_starlight, UCRStarLightLoader
from analysis.feature_engineering.feature_engineering import extract_features_batch
from analysis.df.df_analysis import DFAnalysis, DFOptions, run_df_analysis
from analysis.markov.ssmm_analysis import SSMMAnalysis


def run_baseline_experiment(X_train, y_train, X_test, y_test):
    print("\n" + "="*50)
    print("BASELINE EXPERIMENTS")
    print("="*50)

    results = []
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    }

    for name, classifier in classifiers.items():
        print(f"Training {name} on raw time series...")
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"{name} Accuracy: {acc:.4f}")
        print(f"{name} F1-Score: {f1:.4f}")

        results.append({
            'method': f'Baseline - {name}',
            'accuracy': acc,
            'f1_score': f1,
            'classifier': classifier
        })

    return results


def run_feature_engineering_experiment(X_train, y_train, X_test, y_test):
    print("\n" + "="*50)
    print("FEATURE ENGINEERING EXPERIMENT")
    print("="*50)

    print("Extracting features from training data...")
    train_df = extract_features_batch(X_train)
    print(f"Extracted {train_df.shape[1]} features from {len(X_train)} samples")

    print("Extracting features from test data...")
    test_df = extract_features_batch(X_test)

    print("Training Random Forest on extracted features...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_df, y_train)
    y_pred = rf.predict(test_df)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Feature Engineering Accuracy: {acc:.4f}")
    print(f"Feature Engineering F1-Score: {f1:.4f}")

    results = {
        'method': 'Feature Engineering',
        'accuracy': acc,
        'f1_score': f1,
        'classifier': rf
    }
    return results


def run_df_experiment(X_train, y_train, X_test, y_test):
    print("\n" + "="*50)
    print("DERIVATIVES FIELDS EXPERIMENT")
    print("="*50)

    options = DFOptions()
    options.derivative_order = 1
    options.field_size = 16
    options.normalize_lightcurves = True

    print(f"DF Config: order={options.derivative_order}, field={options.field_size}x{options.field_size}, normalize={options.normalize_lightcurves}")
    df_res = run_df_analysis(X_train, y_train, X_test, y_test, options)

    y_pred = df_res['predictions']
    acc = df_res['accuracy']
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Derivatives Fields Accuracy: {acc:.4f}")
    print(f"Derivatives Fields F1-Score: {f1:.4f}")

    df_res['method'] = 'Derivatives Fields'
    df_res['f1_score'] = f1
    return df_res


def run_ssmm_experiment(X_train, y_train, X_test, y_test):
    print("\n" + "="*50)
    print("SEMI-SUPERVISED MARKOV MODELS EXPERIMENT")
    print("="*50)

    ssmm = SSMMAnalysis(n_states=5, use_hmm=True)
    ssmm.fit(X_train, y_train)
    y_pred = ssmm.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"SSMM Accuracy: {acc:.4f}")
    print(f"SSMM F1-Score: {f1:.4f}")

    return {
        'method': 'SSMM',
        'accuracy': acc,
        'f1_score': f1,
        'classifier': ssmm
    }


def visualize_results(results_list):
    print("\n" + "="*50)
    print("RESULTS COMPARISON")
    print("="*50)

    records = []
    for res in results_list:
        if isinstance(res, list):
            records.extend(res)
        else:
            records.append(res)

    df = pd.DataFrame(records)
    df_sorted = df.sort_values('f1_score', ascending=False)

    print("\nComparison by F1-Score:")
    print(df_sorted[['method','f1_score']].to_string(index=False))

    plt.figure(figsize=(10,6))
    plt.bar(df_sorted['method'], df_sorted['f1_score'], color='skyblue')
    plt.ylabel('F1-Score')
    plt.xlabel('Method')
    plt.title('Method Comparison by F1-Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return df_sorted


def main():
    print("Variable Star Analysis - UCR StarLightCurves Dataset")
    print("="*60)

    DATA_DIR = "data/UCR"
    if not Path(DATA_DIR).exists():
        print(f"Data directory {DATA_DIR} not found")
        return

    loader = UCRStarLightLoader(DATA_DIR)
    X_train, y_train, X_test, y_test = loader.load_data()

    print(f"Loaded {len(X_train)} train, {len(X_test)} test samples, seq len {X_train.shape[1]}")

    all_results = []
    all_results.append(run_baseline_experiment(X_train, y_train, X_test, y_test))
    all_results.append(run_feature_engineering_experiment(X_train, y_train, X_test, y_test))
    all_results.append(run_df_experiment(X_train, y_train, X_test, y_test))
    all_results.append(run_ssmm_experiment(X_train, y_train, X_test, y_test))

    visualize_results(all_results)


if __name__ == "__main__":
    main()