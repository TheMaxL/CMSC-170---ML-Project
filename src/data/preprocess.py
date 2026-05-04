import pandas as pd
import numpy as np
import re
import json
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, config_path="config/config.yaml"):
        """Initialize preprocessor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set random seed for reproducibility
        np.random.seed(self.config['project']['seed'])
        
        # Define column mappings (adjust based on your actual TSV headers)
        self.column_map = {
            'Entry': 'accession',  
            'Entry Name': 'Entry_Name',
            'Protein names': 'Protein_Names', 
            'Gene Names': 'Gene_Names',
            'Length': 'Length',
            'Organism': 'Organism',
            'Subcellular location [CC]': 'Subcellular_Location',
            'Interacts with': 'Interacts_With',
            'Disruption phenotype': 'Disruption_Phenotype',
            'Organism (ID)': 'Taxon_ID',
            'Domain [CC]': 'Domains',
            'Transmembrane': 'Transmembrane_Features'  
        }
        
    def load_data(self, tsv_path):
        """Load TSV and map column names"""
        print(f"Loading data from {tsv_path}...")
        df = pd.read_csv(tsv_path, sep='\t')
        
        # Rename columns for consistency
        df = df.rename(columns=self.column_map)
        print(f"Loaded {len(df):,} proteins with {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
        return df
    
    def extract_tm_count(self, df):
        """Extract target variable: number of transmembrane helices"""
        print("\nExtracting TM helix count (target variable)...")
        
        def count_tm_features(text):
            if pd.isna(text):
                return 0
            # Count TRANSMEM entries in the feature column
            matches = re.findall(r'TRANSMEM', str(text))
            return len(matches)
        
        df['tm_helix_count'] = df['Transmembrane_Features'].apply(count_tm_features)
        
        # Summary statistics
        print(f"TM count distribution:")
        print(df['tm_helix_count'].value_counts().sort_index().head(10))
        print(f"Proteins with 0 TM helices: {(df['tm_helix_count'] == 0).mean():.1%}")
        print(f"Proteins with ≥1 TM helix: {(df['tm_helix_count'] > 0).mean():.1%}")
        print(f"Max TM helices: {df['tm_helix_count'].max()}")
        
        return df
    
    def create_domain_features(self, df):
        """Create binary features for known TM domains"""
        print("\nCreating domain-based features...")
        
        domain_keywords = {
            'gpcr_domain': ['gpcr', 'g protein-coupled', '7 transmembrane', '7tm'],
            'ion_channel_domain': ['ion channel', 'potassium channel', 'sodium channel', 'calcium channel'],
            'transporter_domain': ['transporter', 'solute carrier', 'slc', 'abc transporter'],
            'receptor_domain': ['receptor', 'tyrosine kinase receptor'],
            'enzyme_domain': ['kinase', 'phosphatase', 'synthase', 'dehydrogenase'],
            'structural_domain': ['collagen', 'keratin', 'myosin', 'actin']
        }
        
        for domain_name, keywords in domain_keywords.items():
            df[domain_name] = df['Domains'].fillna('').apply(
                lambda x: 1 if any(kw in str(x).lower() for kw in keywords) else 0
            )
        
        print(f"Added {len(domain_keywords)} domain features")
        for domain in domain_keywords.keys():
            print(f"  {domain}: {(df[domain] == 1).sum():,} proteins ({df[domain].mean():.1%})")
        
        return df
    
    def create_location_features(self, df):
        """Extract subcellular location features"""
        print("\nCreating subcellular location features...")
        
        location_keywords = {
            'plasma_membrane': ['cell membrane', 'plasma membrane'],
            'nucleus': ['nucleus', 'nuclear'],
            'cytoplasm': ['cytoplasm', 'cytosol', 'perinuclear'],
            'mitochondrion': ['mitochondrion', 'mitochondrial'],
            'er': ['endoplasmic reticulum', 'er '],
            'golgi': ['golgi'],
            'secreted': ['secreted', 'extracellular'],
            'lysosome': ['lysosome', 'endosome'],
            'peroxisome': ['peroxisome'],
            'cytoskeleton': ['cytoskeleton', 'microtubule', 'actin']
        }
        
        for loc_name, keywords in location_keywords.items():
            df[loc_name] = df['Subcellular_Location'].fillna('').apply(
                lambda x: 1 if any(kw in str(x).lower() for kw in keywords) else 0
            )
        
        print(f"Added {len(location_keywords)} location features")
        
        return df
    
    def create_interaction_features(self, df):
        """Extract interaction-based features"""
        print("\nCreating interaction features...")
        
        # Count interaction partners
        df['interaction_count'] = df['Interacts_With'].fillna('').apply(
            lambda x: len(str(x).split(';')) if pd.notna(x) and str(x).strip() else 0
        )
        
        # Binary: has any interactions
        df['has_interactions'] = (df['interaction_count'] > 0).astype(int)
        
        print(f"Proteins with interactions: {df['has_interactions'].mean():.1%}")
        print(f"Mean interaction count: {df['interaction_count'].mean():.2f}")
        
        return df
    
    def create_phenotype_features(self, df):
        """Extract disruption phenotype features"""
        print("\nCreating phenotype features...")
        
        phenotype_keywords = {
            'lethal': ['lethal', 'death', 'embryonic lethal'],
            'membrane_defect': ['membrane', 'permeability', 'vesicle'],
            'developmental': ['development', 'morphogenesis', 'differentiation'],
            'neurological': ['brain', 'neuron', 'seizure', 'ataxia'],
            'immune': ['immune', 'inflammation', 'cytokine']
        }
        
        for phen_name, keywords in phenotype_keywords.items():
            df[f'phenotype_{phen_name}'] = df['Disruption_Phenotype'].fillna('').apply(
                lambda x: 1 if any(kw in str(x).lower() for kw in keywords) else 0
            )
        
        print(f"Added {len(phenotype_keywords)} phenotype features")
        
        return df
    
    def create_sequence_features(self, df):
        """Create sequence-based features from length column"""
        print("\nCreating sequence features...")
        
        # Basic length features
        df['log_length'] = np.log1p(df['Length'])  # Log transform for skewed distribution
        df['length_category'] = pd.cut(df['Length'], 
                                        bins=[0, 200, 500, 1000, 10000],
                                        labels=['small', 'medium', 'large', 'very_large'])
        
        print(f"Length statistics: min={df['Length'].min()}, max={df['Length'].max()}, mean={df['Length'].mean():.1f}")
        
        return df
    
    def create_prior_features(self, df):
        """Create Bayesian prior features based on external knowledge"""
        print("\nCreating Bayesian prior features...")
        
        # Prior based on subcellular location
        membrane_prior = {
            'plasma_membrane': 0.8,
            'nucleus': 0.05,
            'cytoplasm': 0.1,
            'mitochondrion': 0.3,
            'er': 0.6,
            'secreted': 0.0
        }
        
        # Calculate location-based prior
        df['location_prior'] = 0.0
        for loc, prior in membrane_prior.items():
            if loc in df.columns:
                df['location_prior'] += df[loc] * prior
    
        # Prior based on domains (GPCR has 7 TM by default)
        df['domain_prior'] = 0.0
        df.loc[df['gpcr_domain'] == 1, 'domain_prior'] = 7.0
        df.loc[df['ion_channel_domain'] == 1, 'domain_prior'] = 4.0
        df.loc[df['transporter_domain'] == 1, 'domain_prior'] = 6.0
        
        # Combined prior (weighted average)
        df['combined_prior'] = (df['location_prior'] + df['domain_prior'] / 10) / 2
        
        print(f"Prior stats: mean={df['combined_prior'].mean():.3f}, std={df['combined_prior'].std():.3f}")
        
        return df
    
    def clean_data(self, df):
        """Handle missing values and outliers"""
        print("\nCleaning data...")
        
        initial_count = len(df)
        
        # Remove rows with missing length (critical)
        df = df.dropna(subset=['Length', 'tm_helix_count'])
        
        # Cap extreme TM counts (99.9th percentile)
        max_tm = df['tm_helix_count'].quantile(0.999)
        df['tm_helix_count'] = df['tm_helix_count'].clip(upper=max_tm)
        
        # Fill remaining NAs with 0 for binary features
        binary_cols = [col for col in df.columns if col.startswith(('gpcr', 'ion_', 'transporter', 'receptor_', 'plasma_', 'has_'))]
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Fill NAs in numeric columns with median
        numeric_cols = ['interaction_count', 'location_prior', 'domain_prior', 'combined_prior']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        print(f"Removed {initial_count - len(df)} rows with critical missing data")
        print(f"Final dataset: {len(df):,} proteins")
        
        return df
    
    def select_features(self, df):
        """Select final feature set for modeling"""
        print("\nSelecting features for ML models...")
        
        # Numerical features
        numerical_features = [
            'Length', 'log_length', 'interaction_count', 'location_prior', 
            'domain_prior', 'combined_prior'
        ]
        
        # Binary/categorical features
        binary_features = [
            'gpcr_domain', 'ion_channel_domain', 'transporter_domain', 
            'receptor_domain', 'plasma_membrane', 'nucleus', 'cytoplasm', 
            'mitochondrion', 'er', 'secreted', 'has_interactions'
        ]
        
        # Make sure all features exist
        numerical_features = [f for f in numerical_features if f in df.columns]
        binary_features = [f for f in binary_features if f in df.columns]
        
        # Combine features
        feature_cols = numerical_features + binary_features
        
        # Target
        target_col = 'tm_helix_count'
        
        print(f"Selected {len(feature_cols)} features:")
        print(f"  Numerical: {len(numerical_features)}")
        print(f"  Binary: {len(binary_features)}")
        
        return df[feature_cols], df[target_col], feature_cols
    
    def split_and_scale(self, X, y, feature_names):
        """Split into train/test and scale numerical features"""
        print("\nSplitting and scaling data...")
        
        # Train/test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.config['project']['seed'], stratify=(y > 0)
        )
        
        print(f"Training set: {len(X_train):,} proteins")
        print(f"Test set: {len(X_test):,} proteins")
        
        # Separate numerical and binary columns
        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        binary_cols = [col for col in feature_names if col not in numerical_cols]
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer([
            ('scaler', StandardScaler(), numerical_cols),
            ('passthrough', 'passthrough', binary_cols)
        ])
        
        # Fit on training, transform both
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_test_scaled = preprocessor.transform(X_test)
        
        # Save the preprocessor for later use
        joblib.dump(preprocessor, 'data/processed/preprocessor.pkl')
        
        return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor
    
    def save_data(self, X_train, X_test, y_train, y_test, feature_names, preprocessor):
        """Save all processed data for teammates"""
        print("\nSaving processed data...")
        
        # Create directories if needed
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
        
        # Save summary statistics for teammates
        summary = {
            'total_samples': len(y_train) + len(y_test),
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'tm_count_mean': float(y_train.mean()),
            'tm_count_std': float(y_train.std()),
            'tm_count_distribution': y_train.value_counts().sort_index().to_dict(),
            'features': feature_names
        }
        
        with open('results/tables/data_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save as CSV for easy viewing
        pd.DataFrame(X_train).to_csv('data/processed/X_train.csv', index=False)
        pd.Series(y_train).to_csv('data/processed/y_train.csv', index=False, header=['tm_helix_count'])
        
        print(f"Saved files to data/processed/")
        print(f"  - X_train.npy ({X_train.shape})")
        print(f"  - X_test.npy ({X_test.shape})")
        print(f"  - y_train.npy, y_test.npy")
        print(f"  - feature_names.json")
        print(f"  - preprocessor.pkl")
    
    def generate_data_dictionary(self, feature_names):
        """Create a data dictionary for teammates"""
        data_dict = {
            'dataset_name': 'UniProt Human Proteins - TM Helix Prediction',
            'target_variable': {
                'name': 'tm_helix_count',
                'description': 'Number of transmembrane helices in the protein',
                'type': 'integer count (0-12 typically)',
                'distribution': 'Zero-inflated (many proteins have 0 TM helices)'
            },
            'features': {}
        }
        
        feature_descriptions = {
            'Length': 'Protein sequence length (number of amino acids)',
            'log_length': 'Log-transformed length (handles skew)',
            'interaction_count': 'Number of known protein interaction partners',
            'location_prior': 'Prior probability of TM helices based on subcellular localization',
            'domain_prior': 'Prior expectation of TM count based on domain family',
            'combined_prior': 'Weighted combination of location and domain priors',
            'gpcr_domain': 'Protein contains GPCR domain (binary)',
            'ion_channel_domain': 'Protein contains ion channel domain (binary)',
            'transporter_domain': 'Protein contains transporter domain (binary)',
            'receptor_domain': 'Protein contains receptor domain (binary)',
            'plasma_membrane': 'Protein localizes to plasma membrane (binary)',
            'nucleus': 'Protein localizes to nucleus (binary)',
            'cytoplasm': 'Protein localizes to cytoplasm (binary)',
            'mitochondrion': 'Protein localizes to mitochondrion (binary)',
            'er': 'Protein localizes to endoplasmic reticulum (binary)',
            'secreted': 'Protein is secreted (binary)',
            'has_interactions': 'Protein has known interaction partners (binary)'
        }
        
        for feat in feature_names:
            data_dict['features'][feat] = {
                'description': feature_descriptions.get(feat, 'No description available'),
                'type': 'numerical' if feat in ['Length', 'log_length', 'interaction_count', 'location_prior', 'domain_prior', 'combined_prior'] else 'binary'
            }
        
        with open('docs/data_dictionary.md', 'w') as f:
            f.write("# Data Dictionary\n\n")
            f.write(f"## Target Variable: `{data_dict['target_variable']['name']}`\n")
            f.write(f"- {data_dict['target_variable']['description']}\n")
            f.write(f"- Type: {data_dict['target_variable']['type']}\n\n")
            f.write("## Features\n\n")
            f.write("| Feature | Type | Description |\n")
            f.write("|---------|------|-------------|\n")
            for feat, info in data_dict['features'].items():
                f.write(f"| {feat} | {info['type']} | {info['description']} |\n")
        
        print("\nGenerated data dictionary: docs/data_dictionary.md")
    
    def run_full_pipeline(self, tsv_path):
        """Execute all preprocessing steps"""
        print("="*60)
        print("DATA PREPROCESSING PIPELINE - Person 1")
        print("="*60)
        
        # Load
        df = self.load_data(tsv_path)
        
        # Extract target
        df = self.extract_tm_count(df)
        
        # Feature engineering
        df = self.create_domain_features(df)
        df = self.create_location_features(df)
        df = self.create_interaction_features(df)
        df = self.create_phenotype_features(df)
        df = self.create_sequence_features(df)
        df = self.create_prior_features(df)
        
        # Clean
        df = self.clean_data(df)
        
        # Select features
        X, y, feature_names = self.select_features(df)
        
        # Split and scale
        X_train, X_test, y_train, y_test, preprocessor = self.split_and_scale(X, y, feature_names)
        
        # Save
        self.save_data(X_train, X_test, y_train, y_test, feature_names, preprocessor)
        self.generate_data_dictionary(feature_names)
        
        print("\n" + "="*60)
        print("✅ DATA PREPROCESSING COMPLETE!")
        print("="*60)
        print("\nFiles ready for teammates:")
        print("  • Person 2 (EDA): Use data/processed/X_train.csv, y_train.csv")
        print("  • Person 3 (Model): Load data/processed/X_train.npy, y_train.npy")
        print("  • Person 4 (Evaluation): Use data/processed/X_test.npy, y_test.npy")
        print("\n📊 Data dictionary: docs/data_dictionary.md")
        
        return X_train, X_test, y_train, y_test, feature_names

if __name__ == "__main__":
    # Run the full pipeline
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, features = preprocessor.run_full_pipeline(
        tsv_path="data/raw/uniprot_human_60,000_entries.tsv"
    )