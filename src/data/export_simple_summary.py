import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path

def create_full_readable_dataset():
    """Create FULL human-readable dataset (all 60k entries)"""
    
    print("="*60)
    print("Creating FULL Human-Readable Dataset")
    print("="*60)
    
    # Load the preprocessor and feature names
    preprocessor = joblib.load('data/processed/preprocessor.pkl')
    
    with open('data/processed/feature_names.json', 'r') as f:
        feature_info = json.load(f)
        feature_names = feature_info['feature_names']
    
    # Load ALL scaled data (train + test combined)
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    # Combine train and test
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    
    print(f"\nCombined dataset: {X_all.shape[0]:,} proteins")
    print(f"Features: {X_all.shape[1]}")
    
    # Get the scaler
    scaler = preprocessor.named_transformers_['scaler']
    
    # Split features (same as in preprocessing)
    numerical_cols = feature_names[:6]
    binary_cols = feature_names[6:]
    
    # Create readable dataframe
    print("\nConverting to readable format (this may take a moment)...")
    readable_data = []
    
    for i in range(X_all.shape[0]):
        row = {}
        
        # Convert numerical features back to original scale
        for j, col in enumerate(numerical_cols):
            scaled_val = X_all[i, j]
            original_val = scaled_val * scaler.scale_[j] + scaler.mean_[j]
            row[col] = round(original_val, 2)
        
        # Binary features (round to 0 or 1)
        for j, col in enumerate(binary_cols):
            idx = len(numerical_cols) + j
            scaled_val = X_all[i, idx]
            row[col] = 1 if scaled_val > 0.5 else 0  # Round to 0/1
        
        # Add target
        row['tm_helix_count'] = int(y_all[i])
        
        readable_data.append(row)
        
        # Progress indicator
        if (i + 1) % 10000 == 0:
            print(f"   Processed {i+1:,}/{X_all.shape[0]:,} proteins...")
    
    # Create DataFrame
    df_readable = pd.DataFrame(readable_data)
    
    # Save as CSV (may be large!)
    output_path = 'results/tables/X_train_readable_full.csv'
    df_readable.to_csv(output_path, index=False)
    
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    
    print(f"\n✅ Saved FULL readable dataset to: {output_path}")
    print(f"   File size: {file_size_mb:.1f} MB")
    print(f"   Shape: {df_readable.shape[0]:,} rows × {df_readable.shape[1]} columns")
    
    # Also save as compressed parquet (smaller file)
    parquet_path = 'results/tables/X_train_readable_full.parquet'
    df_readable.to_parquet(parquet_path, index=False)
    parquet_size_mb = Path(parquet_path).stat().st_size / (1024 * 1024)
    print(f"✅ Saved compressed version to: {parquet_path}")
    print(f"   Compressed size: {parquet_size_mb:.1f} MB")
    
    # Create a metadata file
    metadata = {
        'total_proteins': len(df_readable),
        'features': list(df_readable.columns),
        'tm_helix_distribution': df_readable['tm_helix_count'].value_counts().to_dict(),
        'zero_inflated_percent': float((df_readable['tm_helix_count'] == 0).mean() * 100),
        'mean_tm_helix': float(df_readable['tm_helix_count'].mean()),
        'file_locations': {
            'csv': output_path,
            'parquet': parquet_path
        }
    }
    
    with open('results/tables/readable_dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Saved metadata to: results/tables/readable_dataset_metadata.json")
    
    return df_readable

if __name__ == "__main__":
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    df = create_full_readable_dataset()
    
    print("\n" + "="*60)
    print("PREVIEW OF FIRST 5 ROWS:")
    print("="*60)
    print(df.head().to_string())