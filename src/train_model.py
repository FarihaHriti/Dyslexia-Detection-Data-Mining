import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from data_loader import load_data

def run_experiments():
    print("Loading all datasets...")
    data_map = load_data(".")
    
    etdd_list = data_map['etdd70']
    kron_list = data_map['kronoberg']
    
    print(f"\nLoaded Samples: ETDD70={len(etdd_list)}, Kronoberg={len(kron_list)}")
    
    # helper to df
    def to_df(data):
        if not data: return pd.DataFrame()
        return pd.DataFrame(data)
        
    df_etdd = to_df(etdd_list)
    df_kron = to_df(kron_list)
    
    # Feature columns (Domain-Invariant)
    feature_cols = [
        'fixation_duration_mean', 'fixation_duration_std', 'fixation_duration_max',
        'saccade_length_mean', 'saccade_length_std', 'fix_sac_ratio', 'regression_ratio'
    ]
    
    if df_etdd.empty:
        print("CRITICAL: No ETDD70 data loaded. Cannot run Experiment 1.")
        return

    # Imputer
    imputer = SimpleImputer(strategy='mean')
    
    # --- Experiment I: Intra-Dataset Baseline (ETDD70) ---
    print("\n=== Experiment I: Intra-Dataset Baseline (ETDD70) ===")
    X_etdd = df_etdd[feature_cols].values
    y_etdd = df_etdd['label'].values
    
    # 80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(X_etdd, y_etdd, test_size=0.2, random_state=42)
    
    # Prepare results storage
    results = {
        'Exp1': {},
        'Exp2': {}
    }

    # Models to compare (Optimized via Grid Search and PowerTransformer)
    from sklearn.preprocessing import PowerTransformer
    models = {
        'SVM': make_pipeline(imputer, StandardScaler(), SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)),
        'Random Forest': make_pipeline(imputer, PowerTransformer(method='yeo-johnson'), RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
        'XGBoost': make_pipeline(imputer, PowerTransformer(method='yeo-johnson'), XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.1, subsample=0.8, use_label_encoder=False, eval_metric='logloss', random_state=42))
    }

    for name, clf in models.items():
        print(f"\nTraining {name} for Experiment I...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results['Exp1'][name] = {
            'accuracy': acc,
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
        print(f"{name} Accuracy: {acc}")
    
    # --- Experiment II: Cross-Dataset Generalization (Kronoberg) ---
    print("\n=== Experiment II: Cross-Dataset Generalization (Kronoberg) ===")
    if not df_kron.empty:
        X_kron = df_kron[feature_cols].values
        y_kron = df_kron['label'].values
        
        # Models for Experiment II (Using optimized transfer learning pipelines)
        for name, model_obj in models.items():
            print(f"\nEvaluating {name} for Experiment II...")
            
            # Re-fit on full ETDD for generalization
            model_obj.fit(X_etdd, y_etdd)
            y_pred_kron = model_obj.predict(X_kron)
            
            acc_kron = accuracy_score(y_kron, y_pred_kron)
            results['Exp2'][name] = {
                'accuracy': acc_kron,
                'report': classification_report(y_kron, y_pred_kron, output_dict=True),
                'cm': confusion_matrix(y_kron, y_pred_kron).tolist()
            }
            print(f"{name} Accuracy on Kronoberg: {acc_kron}")

        # Save all results for visualization
        import json
        with open('comparison_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        print("\nComparative results saved to 'comparison_results.json'")
    else:
        print("No Kronoberg data for Experiment II.")

if __name__ == "__main__":
    run_experiments()
