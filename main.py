"""
Main script for running variable star analysis experiments.
Demonstrates usage of all modules with UCR StarLightCurves dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Import our modules
# Import our modules
from lib.data_loader import load_ucr_starlight, UCRStarLightLoader
from analysis.feature_engineering.feature_engineering import LightCurveFeatureExtractor, extract_features_batch
from analysis.df.df_analysis import DFAnalysis, DFOptions, run_df_analysis
from analysis.markov.ssmm_analysis import SSMMAnalysis



def run_feature_engineering_experiment(X_train, y_train, X_test, y_test):
    """
    Run feature engineering experiment.
    """
    print("\\n" + "="*50)
    print("FEATURE ENGINEERING EXPERIMENT")
    print("="*50)
    
    # Extract features
    print("Extracting features from training data...")
    train_features_df = extract_features_batch(X_train)
    print(f"Extracted {train_features_df.shape[1]} features from {len(X_train)} training samples")
    
    print("Extracting features from test data...")
    test_features_df = extract_features_batch(X_test)
    
    # Train classifier on features
    print("Training Random Forest on extracted features...")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(train_features_df, y_train)
    
    # Predict and evaluate
    y_pred = rf_classifier.predict(test_features_df)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Feature Engineering Accuracy: {accuracy:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': train_features_df.columns,
        'importance': rf_classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    return {
        'method': 'Feature Engineering',
        'accuracy': accuracy,
        'classifier': rf_classifier,
        'features': train_features_df,
        'feature_importance': feature_importance
    }


def run_df_experiment(X_train, y_train, X_test, y_test):
    """
    Run Derivatives Fields experiment.
    """
    print("\\n" + "="*50)
    print("DERIVATIVES FIELDS EXPERIMENT")
    print("="*50)
    
    # Configure DF options
    df_options = DFOptions()
    df_options.derivative_order = 1
    df_options.field_size = 16  # Smaller field for faster computation
    df_options.normalize_lightcurves = True
    
    print(f"DF Configuration:")
    print(f"  - Derivative order: {df_options.derivative_order}")
    print(f"  - Field size: {df_options.field_size}x{df_options.field_size}")
    print(f"  - Normalize: {df_options.normalize_lightcurves}")
    
    # Run DF analysis
    print("\\nRunning Derivatives Fields analysis...")
    base_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    df_results = run_df_analysis(X_train, y_train, X_test, y_test, df_options)
    
    print(f"Derivatives Fields Accuracy: {df_results['accuracy']:.4f}")
    
    return df_results


def run_ssmm_experiment(X_train, y_train, X_test, y_test):
    """
    Run Semi-Supervised Markov Models experiment.
    """
    print("\\n" + "="*50)
    print("SEMI-SUPERVISED MARKOV MODELS EXPERIMENT")
    print("="*50)
    
    # Configure SSMM
    print("Training SSMM with HMM components...")
    ssmm_classifier = SSMMAnalysis(n_states=5, use_hmm=True)
    
    # Train
    ssmm_classifier.fit(X_train, y_train)
    
    # Predict
    y_pred = ssmm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"SSMM Accuracy: {accuracy:.4f}")
    
    return {
        'method': 'SSMM',
        'accuracy': accuracy,
        'classifier': ssmm_classifier,
        'predictions': y_pred
    }


def run_baseline_experiment(X_train, y_train, X_test, y_test):
    """
    Run baseline experiments with simple classifiers.
    """
    print("\\n" + "="*50)
    print("BASELINE EXPERIMENTS")
    print("="*50)
    
    results = []
    
    # Raw time series with different classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
    }
    
    for name, classifier in classifiers.items():
        print(f"Training {name} on raw time series...")
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        results.append({
            'method': f'Baseline - {name}',
            'accuracy': accuracy,
            'classifier': classifier
        })
    
    return results


def visualize_results(results_list):
    """
    Visualize comparison of all methods.
    """
    print("\\n" + "="*50)
    print("RESULTS COMPARISON")
    print("="*50)
    
    # Extract results
    methods = []
    accuracies = []
    
    for results in results_list:
        if isinstance(results, list):
            for result in results:
                methods.append(result['method'])
                accuracies.append(result['accuracy'])
        else:
            methods.append(results['method'])
            accuracies.append(results['accuracy'])
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Method': methods,
        'Accuracy': accuracies
    }).sort_values('Accuracy', ascending=False)
    
    print("\\nAccuracy Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(methods)), accuracies)
    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.title('Variable Star Classification - Method Comparison')
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return comparison_df


def main():
    """
    Main function to run all experiments.
    """
    print("Variable Star Analysis - UCR StarLightCurves Dataset")
    print("="*60)
    
    # Configuration
    DATA_DIR = "data/UCR"  # Update this path
    
    # Check if data directory exists
    if not Path(DATA_DIR).exists():
        print(f"Error: Data directory {DATA_DIR} not found!")
        print("Please download UCR StarLightCurves dataset and update DATA_DIR path.")
        return
    
    # Load dataset
    print("Loading UCR StarLightCurves dataset...")
    try:
        X_train, y_train, X_test, y_test = load_ucr_starlight(DATA_DIR, normalize=True)
        print(f"Loaded {len(X_train)} training and {len(X_test)} test samples")
        print(f"Time series length: {X_train.shape[1]}")
        print(f"Classes: {np.unique(y_train)}")
        
        # Dataset statistics
        loader = UCRStarLightLoader(DATA_DIR)
        stats = loader.get_dataset_stats()
        class_info = loader.get_class_info()
        
        print("\\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
        print("\\nClass Information:")
        for class_id, class_name in class_info.items():
            print(f"  {class_id}: {class_name}")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check the data directory and file format.")
        return
    
    # Run experiments
    all_results = []
    
    try:
        # Baseline experiments
        baseline_results = run_baseline_experiment(X_train, y_train, X_test, y_test)
        all_results.append(baseline_results)
        
        # Feature engineering experiment
        fe_results = run_feature_engineering_experiment(X_train, y_train, X_test, y_test)
        all_results.append(fe_results)
        
        # Derivatives Fields experiment
        df_results = run_df_experiment(X_train, y_train, X_test, y_test)
        all_results.append(df_results)
        
        # SSMM experiment
        ssmm_results = run_ssmm_experiment(X_train, y_train, X_test, y_test)
        all_results.append(ssmm_results)
        
    except Exception as e:
        print(f"Error during experiments: {e}")
        import traceback
        traceback.print_exc()
    
    # Visualize results
    if all_results:
        comparison_df = visualize_results(all_results)
        
        # Save results
        comparison_df.to_csv('results/method_comparison.csv', index=False)
        print(f"\\nResults saved to 'results/method_comparison.csv'")


if __name__ == "__main__":
    main()