import pandas as pd
import numpy as np
import re
from pathlib import Path

def load_and_preview_original_data():
    """Load the original TSV and create readable summaries"""
    
    print("="*60)
    print("Creating Human-Readable Data Summary")
    print("="*60)
    
    # Load original TSV
    tsv_path = 'data/raw/uniprot_human_60,000_entries.tsv'
    df = pd.read_csv(tsv_path, sep='\t')
    
    print(f"\nLoaded {len(df):,} proteins")
    print(f"Columns: {list(df.columns)}")
    
    # Extract TM count from Transmembrane column
    def count_tm(text):
        if pd.isna(text):
            return 0
        return len(re.findall(r'TRANSMEM', str(text)))
    
    df['tm_helix_count'] = df['Transmembrane'].apply(count_tm)
    
    # Create readable summary
    print("\n" + "="*60)
    print("TARGET VARIABLE: TM Helix Distribution")
    print("="*60)
    
    tm_dist = df['tm_helix_count'].value_counts().sort_index()
    print(f"\nTM Count Distribution:")
    for count, num_proteins in tm_dist.head(15).items():
        pct = num_proteins / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {count:2d} TM: {num_proteins:6,} proteins ({pct:5.1f}%) {bar}")
    
    print(f"\nSummary Statistics:")
    print(f"  Mean: {df['tm_helix_count'].mean():.2f}")
    print(f"  Median: {df['tm_helix_count'].median():.0f}")
    print(f"  Std Dev: {df['tm_helix_count'].std():.2f}")
    print(f"  Range: {df['tm_helix_count'].min()} - {df['tm_helix_count'].max()}")
    print(f"  Zero-inflated: {(df['tm_helix_count'] == 0).mean() * 100:.1f}%")
    
    # Protein length analysis
    print("\n" + "="*60)
    print("NUMERICAL FEATURES")
    print("="*60)
    
    print(f"\nLength (amino acids):")
    print(f"  Mean: {df['Length'].mean():.1f}")
    print(f"  Median: {df['Length'].median():.0f}")
    print(f"  Std Dev: {df['Length'].std():.1f}")
    print(f"  Range: {df['Length'].min()} - {df['Length'].max()}")
    
    # Domain detection (from Domain [CC] column)
    print("\n" + "="*60)
    print("DOMAIN FEATURES (from Domain [CC] column)")
    print("="*60)
    
    domain_keywords = {
        'GPCR': ['gpcr', 'g protein-coupled', '7 transmembrane receptor', '7tm'],
        'Ion Channel': ['ion channel', 'potassium channel', 'sodium channel', 'calcium channel'],
        'Transporter': ['transporter', 'solute carrier', 'slc', 'abc transporter', 'major facilitator'],
        'Kinase': ['kinase', 'phosphoryl', 'kinase domain'],
        'Receptor': ['receptor', 'receptor domain']
    }
    
    for domain_name, keywords in domain_keywords.items():
        mask = df['Domain [CC]'].fillna('').str.lower().str.contains('|'.join(keywords))
        count = mask.sum()
        pct = count / len(df) * 100
        print(f"  {domain_name:15s}: {count:6,} proteins ({pct:5.1f}%)")
    
    # Location extraction (from Subcellular location [CC] column)
    print("\n" + "="*60)
    print("SUBCELLULAR LOCATION FEATURES")
    print("="*60)
    
    location_keywords = {
        'Plasma membrane': ['cell membrane', 'plasma membrane'],
        'Nucleus': ['nucleus', 'nuclear'],
        'Cytoplasm': ['cytoplasm', 'cytosol'],
        'Mitochondrion': ['mitochondrion', 'mitochondrial'],
        'Endoplasmic reticulum': ['endoplasmic reticulum', 'er '],
        'Golgi': ['golgi'],
        'Secreted': ['secreted', 'extracellular'],
        'Lysosome': ['lysosome', 'endosome'],
        'Peroxisome': ['peroxisome']
    }
    
    for loc_name, keywords in location_keywords.items():
        mask = df['Subcellular location [CC]'].fillna('').str.lower().str.contains('|'.join(keywords))
        count = mask.sum()
        pct = count / len(df) * 100
        print(f"  {loc_name:20s}: {count:6,} proteins ({pct:5.1f}%)")
    
    # Interaction analysis
    print("\n" + "="*60)
    print("INTERACTION FEATURES")
    print("="*60)
    
    df['interaction_count'] = df['Interacts with'].fillna('').apply(
        lambda x: len(str(x).split(';')) if pd.notna(x) and str(x).strip() else 0
    )
    
    print(f"  Interactions per protein:")
    print(f"    Mean: {df['interaction_count'].mean():.2f}")
    print(f"    Median: {df['interaction_count'].median():.0f}")
    print(f"    Max: {df['interaction_count'].max()}")
    print(f"  Proteins with any interactions: {(df['interaction_count'] > 0).mean() * 100:.1f}%")
    
    # Create sample of readable data
    print("\n" + "="*60)
    print("CREATING READABLE SAMPLE")
    print("="*60)
    
    # Select columns for readable output
    readable_cols = ['Entry', 'Entry Name', 'Length', 'tm_helix_count', 
                      'Subcellular location [CC]', 'Domain [CC]']
    
    readable_sample = df[readable_cols].head(20).copy()
    
    # Truncate long text
    for col in ['Subcellular location [CC]', 'Domain [CC]']:
        readable_sample[col] = readable_sample[col].fillna('').str[:80]
    
    readable_sample.to_csv('results/tables/readable_sample.csv', index=False)
    print(f"✅ Saved readable sample (20 proteins) to: results/tables/readable_sample.csv")
    
    # Save full summary
    summary = {
        'dataset_size': len(df),
        'tm_helix_statistics': {
            'mean': float(df['tm_helix_count'].mean()),
            'median': float(df['tm_helix_count'].median()),
            'std': float(df['tm_helix_count'].std()),
            'min': int(df['tm_helix_count'].min()),
            'max': int(df['tm_helix_count'].max()),
            'zero_inflated_pct': float((df['tm_helix_count'] == 0).mean() * 100),
            'distribution': {int(k): int(v) for k, v in tm_dist.to_dict().items()}
        },
        'length_statistics': {
            'mean': float(df['Length'].mean()),
            'median': float(df['Length'].median()),
            'std': float(df['Length'].std()),
            'min': int(df['Length'].min()),
            'max': int(df['Length'].max())
        }
    }
    
    with open('results/tables/raw_data_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Saved summary to: results/tables/raw_data_summary.json")
    
    return df