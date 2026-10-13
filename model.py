#!/usr/bin/env python3
"""
AKI Detection Model
CW 1 - SWEMLS

This script trains an XGBoost classifier to predict Acute Kidney Injury (AKI)
based on patient history. It extracts clinical features (recent vs baseline creatinine),
optimizes the decision threshold for the F3 metric, and generates predictions.
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score

# Feature Engineering
def extract_features(df):
    """
    Transforms raw longitudinal data into fixed-length feature vectors.
    
    Features extracted:
    - Demographics: Age, Sex
    - Current Status: Last creatinine result
    - Recent History (0-7 days): Min, Mean
    - Baseline History (8-365 days): Median, Max
    """
    # Normalize sex to numeric (Female=1, Male=0). 
    # Using map handles 'f'/'m' correctly; fillna handles missing data safely.
    df['is_female'] = df['sex'].map({'f': 1, 'm': 0}).fillna(0)
    
    feature_rows = []
    
    for _, row in df.iterrows():
        history = []
        
        # Scan up to 100 columns (safety limit based on CSV structure)
        for i in range(100):
            date_col = f'creatinine_date_{i}'
            result_col = f'creatinine_result_{i}'
            
            # Stop if column doesn't exist or data is missing
            if date_col not in df.columns or pd.isna(row[date_col]) or row[date_col] == '':
                break
            
            try:
                dt = pd.to_datetime(row[date_col])
                val = float(row[result_col])
                history.append({'date': dt, 'val': val})
            except (ValueError, TypeError):
                # Skip individual malformed entries without crashing
                continue
        
        # Initialize Features (Default to NaN for missing history)
        # XGBoost handles NaN values natively as "missing information"
        feats = {
            'age': row['age'],
            'is_female': row['is_female'],
            'last_res': np.nan,
            'min_7d': np.nan,
            'mean_7d': np.nan,
            'median_baseline': np.nan,
            'max_baseline': np.nan
        }

        if history:
            # Sort by date to ensure correct timeline (Oldest -> Newest)
            history.sort(key=lambda x: x['date'])
            
            # Identify Index Test (The most recent result)
            index_test = history[-1]
            index_date = index_test['date']
            feats['last_res'] = index_test['val']
            
            # Buckets for history windows
            recent_vals = []   # 0 < t <= 7 days
            baseline_vals = [] # 8 <= t <= 365 days
            
            # Iterate through previous tests (excluding the current index test)
            for test in history[:-1]:
                diff = index_date - test['date']
                days_ago = diff.total_seconds() / 86400.0
                
                # Filter out future dates (dirty data protection)
                if days_ago < 0: 
                    continue

                if days_ago <= 7:
                    recent_vals.append(test['val'])
                elif 8 <= days_ago <= 365:
                    baseline_vals.append(test['val'])
            
            # Calculate Statistics
            if recent_vals:
                feats['min_7d'] = np.min(recent_vals)
                feats['mean_7d'] = np.mean(recent_vals)
            
            if baseline_vals:
                feats['median_baseline'] = np.median(baseline_vals)
                feats['max_baseline'] = np.max(baseline_vals)

        feature_rows.append(feats)

    return pd.DataFrame(feature_rows)

# Pipeline Logic
def run_pipeline(train_path, input_path, output_path):
    # 1. Load Training Data
    if not os.path.exists(train_path):
        print(f"Error: Training file not found at {train_path}")
        sys.exit(1)
        
    print(f"Loading training data from {train_path}...")
    try:
        train_df = pd.read_csv(train_path)
    except Exception as e:
        print(f"Error reading training CSV: {e}")
        sys.exit(1)
    
    # Target Mapping: 'y' -> 1, 'n' -> 0
    y = train_df['aki'].map({'y': 1, 'n': 0}).astype(int)
    X = extract_features(train_df)
    
    # 2. Threshold Optimization (Cross-Validation)
    print("Optimizing decision threshold for F3 score...")
    
    # XGBoost Config: scale_pos_weight helps with class imbalance
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=3, 
        n_jobs=-1,
        random_state=42
    )
    
    # Stratified K-Fold ensures every fold has a representative mix of AKI cases
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_thresholds = []
    cv_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
        
        model.fit(X_train_fold, y_train_fold)
        probs = model.predict_proba(X_val_fold)[:, 1]
        
        # Grid search for the best threshold on this validation fold
        best_f3 = 0
        best_t = 0.5
        for t in np.arange(0.1, 0.9, 0.02):
            preds = (probs >= t).astype(int)
            # F3 Score: Recall is weighted 3x higher than Precision
            score = fbeta_score(y_val_fold, preds, beta=3)
            if score > best_f3:
                best_f3 = score
                best_t = t
        
        best_thresholds.append(best_t)
        cv_scores.append(best_f3)

    # Use the average of optimal thresholds found during CV
    optimal_threshold = np.mean(best_thresholds)
    avg_f3 = np.mean(cv_scores)
    
    print(f"Verification Complete. Estimated F3 Score: {avg_f3:.4f}")
    print(f"Selected Decision Threshold: {optimal_threshold:.4f}")

    # 3. Final Prediction
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        sys.exit(1)

    print(f"Processing input file: {input_path}")
    try:
        test_df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        sys.exit(1)
        
    X_test = extract_features(test_df)
    
    # Retrain on full dataset
    model.fit(X, y)
    
    # Predict using the optimized threshold
    test_probs = model.predict_proba(X_test)[:, 1]
    final_preds = (test_probs >= optimal_threshold).astype(int)
    
    # Write output matching strict format requirements (single 'aki' column, y/n)
    output_df = pd.DataFrame()
    output_df['aki'] = np.where(final_preds == 1, 'y', 'n')
    
    try:
        output_df.to_csv(output_path, index=False)
        print(f"Predictions written to {output_path}")
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)

# Main Entry Point
def main():
    parser = argparse.ArgumentParser(description="AKI Prediction Model")
    
    # Flags for input, output and training data
    parser.add_argument("--input", default="/data/test.csv", help="Path to input CSV")
    parser.add_argument("--output", default="/data/aki.csv", help="Path to output CSV")
    parser.add_argument("--train", default="/data/training.csv", help="Path to training CSV")
    
    args = parser.parse_args()
    
    run_pipeline(args.train, args.input, args.output)

if __name__ == "__main__":
    main()
