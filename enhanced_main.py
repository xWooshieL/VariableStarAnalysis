"""
Enhanced main script with LM3L and optimized DF/SSMM methods.
Demonstrates state-of-the-art variable star classification.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our modules
from lib.data_loader import load_ucr_starlight, UCRStarLightLoader
from analysis.feature_engineering.feature_engineering import extract_features_batch
from analysis.df.df_analysis import DFAnalysis, DFOptions, run_df_analysis

# Import enhanced modules
try:
    from analysis.df.enhanced_df_analysis import run_enhanced_df_analysis
    from analysis.markov.enhanced_ssmm_analysis import run_enhanced_ssmm_analysis
    from analysis.markov.lm3l import LM3L, create_multiview_features
    from analysis.markov.lm3l_mv import LM3LMV
    ENHANCED_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced modules not available: {e}")
    ENHANCED_MODULES_AVAILABLE = False
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

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': train_df.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 most important features:")
    print(feature_importance.head(10))

    return {
        'method': 'Feature Engineering',
        'accuracy': acc,
        'f1_score': f1,
        'classifier': rf,
        'feature_importance': feature_importance
    }


def run_enhanced_df_experiment(X_train, y_train, X_test, y_test):
    """Run enhanced DF experiment with parameter optimization."""
    if not ENHANCED_MODULES_AVAILABLE:
        print("Enhanced DF module not available, using basic version")
        return run_basic_df_experiment(X_train, y_train, X_test, y_test)
    
    print("\n" + "="*50)
    print("ENHANCED DERIVATIVES FIELDS EXPERIMENT")
    print("="*50)
    
    return run_enhanced_df_analysis(X_train, y_train, X_test, y_test, optimize_params=True)


def run_enhanced_ssmm_experiment(X_train, y_train, X_test, y_test):
    """Run enhanced SSMM experiment with parameter optimization."""
    if not ENHANCED_MODULES_AVAILABLE:
        print("Enhanced SSMM module not available, using basic version")
        return run_basic_ssmm_experiment(X_train, y_train, X_test, y_test)
    
    print("\n" + "="*50)
    print("ENHANCED SSMM EXPERIMENT")
    print("="*50)
    
    return run_enhanced_ssmm_analysis(X_train, y_train, X_test, y_test, optimize_params=True)


def run_lm3l_experiment(X_train, y_train, X_test, y_test):
    """Run LM3L multi-view experiment."""
    if not ENHANCED_MODULES_AVAILABLE:
        print("LM3L module not available, skipping")
        return None
    
    print("\n" + "="*50)
    print("LM3L MULTI-VIEW EXPERIMENT")
    print("="*50)
    
    try:
        # Extract different types of features for multi-view learning
        print("Extracting multi-view features...")
        
        # Statistical features
        stat_features_train = extract_features_batch(X_train)
        stat_features_test = extract_features_batch(X_test)
        
        # Basic DF features (without optimization for speed)
        from analysis.df.df_analysis import DFAnalysis, DFOptions
        basic_df_options = DFOptions()
        basic_df_options.field_size = 16
        basic_df_options.optimize_parameters = False
        
        basic_df = DFAnalysis(options=basic_df_options)
        basic_df.fit(X_train, y_train)
        df_features_train = basic_df.extract_df_features(X_train)
        df_features_test = basic_df.extract_df_features(X_test)
        
        # Create multi-view feature dictionary
        multiview_train = {
            'statistical': stat_features_train.values,
            'derivatives_fields': df_features_train,
            'raw_normalized': (X_train - X_train.mean(axis=1, keepdims=True)) / X_train.std(axis=1, keepdims=True)
        }
        
        multiview_test = {
            'statistical': stat_features_test.values,
            'derivatives_fields': df_features_test,
            'raw_normalized': (X_test - X_test.mean(axis=1, keepdims=True)) / X_test.std(axis=1, keepdims=True)
        }
        
        # Train LM3L classifier
        print("Training LM3L classifier...")
        lm3l = LM3L(
            alpha=1.0,
            beta=5.0,
            gamma=0.1,
            k_neighbors=5,
            max_iter=20,  # Reduced for faster training
            convergence_threshold=1e-4
        )
        
        lm3l.fit(multiview_train, y_train)
        
        # Predict
        y_pred = lm3l.predict(multiview_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"LM3L Accuracy: {acc:.4f}")
        print(f"LM3L F1-Score: {f1:.4f}")
        
        return {
            'method': 'LM3L Multi-View',
            'accuracy': acc,
            'f1_score': f1,
            'classifier': lm3l,
            'predictions': y_pred
        }
        
    except Exception as e:
        print(f"LM3L experiment failed: {e}")
        return None


def run_lm3l_mv_experiment(X_train, y_train, X_test, y_test):
    """Run LM3L-MV matrix-variate experiment."""
    if not ENHANCED_MODULES_AVAILABLE:
        print("LM3L-MV module not available, skipping")
        return None
    
    print("\n" + "="*50)
    print("LM3L-MV MATRIX-VARIATE EXPERIMENT")
    print("="*50)
    
    try:
        # Prepare matrix-variate data
        print("Preparing matrix-variate features...")
        
        # Statistical features (vector)
        stat_features_train = extract_features_batch(X_train).values
        stat_features_test = extract_features_batch(X_test).values
        
        # DF fields (matrix)
        from analysis.df.df_analysis import DFAnalysis, DFOptions
        df_options = DFOptions()
        df_options.field_size = 12  # Smaller for faster computation
        
        df_analyzer = DFAnalysis(options=df_options)
        df_analyzer.fit(X_train, y_train)
        
        # Generate DF matrices
        df_matrices_train = []
        df_matrices_test = []
        
        for lc in X_train:
            df_field = df_analyzer.df_generator.generate_df_field(
                df_analyzer._preprocess_lightcurve(lc)
            )
            df_matrices_train.append(df_field)
        
        for lc in X_test:
            df_field = df_analyzer.df_generator.generate_df_field(
                df_analyzer._preprocess_lightcurve(lc)
            )
            df_matrices_test.append(df_field)
        
        df_matrices_train = np.array(df_matrices_train)
        df_matrices_test = np.array(df_matrices_test)
        
        # Create multi-view matrix data
        multiview_train_mv = {
            'statistical': stat_features_train,
            'df_matrices': df_matrices_train
        }
        
        multiview_test_mv = {
            'statistical': stat_features_test,
            'df_matrices': df_matrices_test
        }
        
        # Train LM3L-MV classifier
        print("Training LM3L-MV classifier...")
        lm3l_mv = LM3LMV(
            alpha=0.5,
            beta=0.5,
            gamma=0.5,
            k_neighbors=5,
            max_iter=15,  # Reduced for faster training
            convergence_threshold=1e-3
        )
        
        lm3l_mv.fit(multiview_train_mv, y_train)
        
        # Predict
        y_pred = lm3l_mv.predict(multiview_test_mv)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"LM3L-MV Accuracy: {acc:.4f}")
        print(f"LM3L-MV F1-Score: {f1:.4f}")
        
        return {
            'method': 'LM3L-MV Matrix-Variate',
            'accuracy': acc,
            'f1_score': f1,
            'classifier': lm3l_mv,
            'predictions': y_pred
        }
        
    except Exception as e:
        print(f"LM3L-MV experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_basic_df_experiment(X_train, y_train, X_test, y_test):
    """Fallback basic DF experiment."""
    print("\n" + "="*50)
    print("BASIC DERIVATIVES FIELDS EXPERIMENT")
    print("="*50)
    
    options = DFOptions()
    options.derivative_order = 1
    options.field_size = 16
    options.normalize_lightcurves = True
    
    df_results = run_df_analysis(X_train, y_train, X_test, y_test, options)
    
    y_pred = df_results['predictions']
    acc = df_results['accuracy']
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Basic DF Accuracy: {acc:.4f}")
    print(f"Basic DF F1-Score: {f1:.4f}")
    
    df_results['method'] = 'Basic Derivatives Fields'
    df_results['f1_score'] = f1
    return df_results


def run_basic_ssmm_experiment(X_train, y_train, X_test, y_test):
    """Fallback basic SSMM experiment."""
    print("\n" + "="*50)
    print("BASIC SSMM EXPERIMENT")
    print("="*50)
    
    ssmm = SSMMAnalysis(n_states=5, use_hmm=True)
    ssmm.fit(X_train, y_train)
    y_pred = ssmm.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Basic SSMM Accuracy: {acc:.4f}")
    print(f"Basic SSMM F1-Score: {f1:.4f}")
    
    return {
        'method': 'Basic SSMM',
        'accuracy': acc,
        'f1_score': f1,
        'classifier': ssmm,
        'predictions': y_pred
    }


def visualize_results(results_list):
    print("\n" + "="*50)
    print("RESULTS COMPARISON")
    print("="*50)

    records = []
    for res in results_list:
        if res is None:
            continue
        if isinstance(res, list):
            records.extend(res)
        else:
            records.append(res)

    if len(records) == 0:
        print("No results to visualize")
        return None

    df = pd.DataFrame(records)
    df_sorted = df.sort_values('f1_score', ascending=False)

    print("\nF1-Score Ranking:")
    for _, row in df_sorted.iterrows():
        print(f"  {row['method']:30} F1: {row['f1_score']:.4f}")

    # Create results plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(df_sorted)), df_sorted['f1_score'])
    plt.xlabel('Method')
    plt.ylabel('F1-Score')
    plt.title('Variable Star Classification - Enhanced Methods Comparison')
    plt.xticks(range(len(df_sorted)), df_sorted['method'], rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, f1) in enumerate(zip(bars, df_sorted['f1_score'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{f1:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return df_sorted


def main():
    print("Enhanced Variable Star Analysis with LM3L Methods")
    print("="*60)

    DATA_DIR = "data/UCR"
    if not Path(DATA_DIR).exists():
        print(f"Data directory {DATA_DIR} not found")
        return

    # Load dataset
    print("Loading UCR StarLightCurves dataset...")
    loader = UCRStarLightLoader(DATA_DIR)
    X_train, y_train, X_test, y_test = loader.load_data()
    
    # Normalize data
    X_train = (X_train - X_train.mean(axis=1, keepdims=True)) / X_train.std(axis=1, keepdims=True)
    X_test = (X_test - X_test.mean(axis=1, keepdims=True)) / X_test.std(axis=1, keepdims=True)

    print(f"Loaded {len(X_train)} train, {len(X_test)} test samples")
    print(f"Time series length: {X_train.shape[1]}")
    print(f"Classes: {np.unique(y_train)}")

    # Run all experiments
    all_results = []
    
    # Basic experiments
    all_results.append(run_baseline_experiment(X_train, y_train, X_test, y_test))
    all_results.append(run_feature_engineering_experiment(X_train, y_train, X_test, y_test))
    
    # Enhanced methods
    all_results.append(run_enhanced_df_experiment(X_train, y_train, X_test, y_test))
    all_results.append(run_enhanced_ssmm_experiment(X_train, y_train, X_test, y_test))
    
    # Advanced multi-view methods
    if ENHANCED_MODULES_AVAILABLE:
        all_results.append(run_lm3l_experiment(X_train, y_train, X_test, y_test))
        all_results.append(run_lm3l_mv_experiment(X_train, y_train, X_test, y_test))
    
    # Visualize and save results
    comparison_df = visualize_results(all_results)
    
    if comparison_df is not None:
        # Ensure results directory exists
        Path('results').mkdir(exist_ok=True)
        comparison_df.to_csv('results/enhanced_method_comparison.csv', index=False)
        print(f"\nResults saved to 'results/enhanced_method_comparison.csv'")
        
        # Print detailed classification report for best method
        best_result = None
        best_f1 = 0.0
        for res_group in all_results:
            if res_group is None:
                continue
            if isinstance(res_group, list):
                for res in res_group:
                    if res['f1_score'] > best_f1:
                        best_f1 = res['f1_score']
                        best_result = res
            else:
                if res_group['f1_score'] > best_f1:
                    best_f1 = res_group['f1_score']
                    best_result = res_group
        
        if best_result and 'predictions' in best_result:
            print(f"\n\nDetailed Classification Report for Best Method ({best_result['method']}):")
            print("="*60)
            
            class_names = ['Cepheid', 'RR Lyrae', 'Eclipsing Binary']
            report = classification_report(y_test, best_result['predictions'], 
                                         target_names=class_names)
            print(report)


if __name__ == "__main__":
    main()