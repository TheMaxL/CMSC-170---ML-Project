import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, config_path="config/config.yaml"):
        """Initialize preprocessor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set random seed for reproducibility
        np.random.seed(self.config['project']['seed'])
        
    def load_data(self, csv_path):
        """Load pre-processed CSV file"""
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)  # Use default comma separator
        print(f"Loaded {len(df):,} proteins with {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        return df
    
    def prepare_features_and_target(self, df):
        """Separate features and target from pre-computed data"""
        print("\nPreparing features and target...")
        
        # The target is already 'tm_count' in your data
        target_col = 'tm_count'
        
        # Define which columns to use as features (exclude metadata and target)
        exclude_cols = ['accession', 'id', 'gene_name', 'protein_name', 'length', 'tm_count']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target_col]
        
        print(f"Target: {target_col}")
        print(f"Target distribution:\n{y.value_counts().sort_index().head(10)}")
        print(f"Proteins with 0 TM: {(y == 0).mean():.1%}")
        print(f"Proteins with ≥1 TM: {(y > 0).mean():.1%}")
        print(f"\nFeatures: {len(feature_cols)} columns")
        print(f"Feature types:")
        print(f"  Numerical: {len(X.select_dtypes(include=['int64', 'float64']).columns)}")
        print(f"  Other: {len(X.select_dtypes(exclude=['int64', 'float64']).columns)}")
        
        return X, y, feature_cols
    
    def split_and_scale(self, X, y, feature_names):
        """Split into train/test and scale numerical features"""
        print("\nSplitting and scaling data...")
        
        # For regression, stratify by binned TM counts to maintain distribution
        y_binned = pd.cut(y, bins=[0, 1, 2, 5, 10, 100], labels=[0, 1, 2, 3, 4])
        
        # Train/test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=self.config['project']['seed'],
            stratify=y_binned
        )
        
        print(f"Training set: {len(X_train):,} proteins")
        print(f"Test set: {len(X_test):,} proteins")
        print(f"Training TM count mean: {y_train.mean():.2f}")
        print(f"Test TM count mean: {y_test.mean():.2f}")
        
        # Identify numerical columns
        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Create scaler for numerical features only
        preprocessor = ColumnTransformer([
            ('scaler', StandardScaler(), numerical_cols)
        ], remainder='passthrough')
        
        # Fit on training, transform both
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_test_scaled = preprocessor.transform(X_test)
        
        # Save the preprocessor
        Path('data/processed').mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, 'data/processed/preprocessor.pkl')
        
        return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor
    
    def save_data(self, X_train, X_test, y_train, y_test, feature_names):
        """Save all processed data"""
        print("\nSaving processed data...")
        
        Path('data/processed').mkdir(parents=True, exist_ok=True)
        Path('results/tables').mkdir(parents=True, exist_ok=True)
        
        # Save numpy arrays
        np.save('data/processed/X_train.npy', X_train)
        np.save('data/processed/X_test.npy', X_test)
        np.save('data/processed/y_train.npy', y_train.values)
        np.save('data/processed/y_test.npy', y_test.values)
        
        # Save feature names
        with open('data/processed/feature_names.json', 'w') as f:
            json.dump({
                'feature_names': feature_names,
                'n_features': len(feature_names),
                'train_shape': X_train.shape,
                'test_shape': X_test.shape
            }, f, indent=2)
        
        # Save summary statistics
        summary = {
            'total_samples': len(y_train) + len(y_test),
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'tm_count_mean': float(y_train.mean()),
            'tm_count_std': float(y_train.std()),
            'tm_count_distribution': y_train.value_counts().sort_index().to_dict(),
            'tm_count_percentiles': {
                '50%': float(y_train.quantile(0.5)),
                '75%': float(y_train.quantile(0.75)),
                '90%': float(y_train.quantile(0.9)),
                '95%': float(y_train.quantile(0.95)),
                '99%': float(y_train.quantile(0.99))
            },
            'features': feature_names
        }
        
        with open('results/tables/data_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save CSV versions for easy viewing
        pd.DataFrame(X_train).to_csv('data/processed/X_train.csv', index=False)
        pd.Series(y_train).to_csv('data/processed/y_train.csv', index=False, header=['tm_helix_count'])
        
        print(f"Saved files to data/processed/")
        print(f"  - X_train.npy ({X_train.shape})")
        print(f"  - X_test.npy ({X_test.shape})")
        print(f"  - y_train.npy, y_test.npy")
        print(f"  - feature_names.json")
        print(f"  - preprocessor.pkl")
    
    def run_full_pipeline(self, csv_path):
        """Execute preprocessing pipeline for already-feature-engineered data"""
        print("="*60)
        print("DATA PREPROCESSING PIPELINE - Person 1")
        print("="*60)
        
        # Load pre-computed data
        df = self.load_data(csv_path)
        
        # Prepare features and target
        X, y, feature_names = self.prepare_features_and_target(df)
        
        # Split and scale
        X_train, X_test, y_train, y_test, preprocessor = self.split_and_scale(X, y, feature_names)
        
        # Save
        self.save_data(X_train, X_test, y_train, y_test, feature_names)
        
        print("\n" + "="*60)
        print("✅ DATA PREPROCESSING COMPLETE!")
        print("="*60)
        print("\nFiles ready for teammates:")
        print("  • Person 2 (EDA): Use data/processed/X_train.csv, y_train.csv")
        print("  • Person 3 (Model): Load data/processed/X_train.npy, y_train.npy")
        print("  • Person 4 (Evaluation): Use data/processed/X_test.npy, y_test.npy")
        
        return X_train, X_test, y_train, y_test, feature_names

if __name__ == "__main__":
    # Run the full pipeline
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, features = preprocessor.run_full_pipeline(
        csv_path="data/raw/uniprot_tm_positive_200,000_with_features.csv"
    )