import requests
import csv
import time
import math
import re
from collections import Counter

def count_tm_helices(transmembrane_value):
    """Extract TM helix count from Transmembrane field"""
    if not transmembrane_value or transmembrane_value == '':
        return 0
    
    # Transmembrane field format examples:
    # "TRANSMEM 23..45" or "TRANSMEM 23..45; TRANSMEM 67..89"
    # Count the number of TRANSMEM entries
    tm_count = transmembrane_value.upper().count('TRANSMEM')
    
    # If no "TRANSMEM" keyword, try counting semicolons + 1
    if tm_count == 0 and ';' in transmembrane_value:
        tm_count = transmembrane_value.count(';') + 1
    
    return tm_count

def compute_features(row, headers):
    """Compute all features for a protein entry"""
    # Create header to index mapping
    idx = {header: i for i, header in enumerate(headers)}
    
    features = {}
    
    # Numerical features
    try:
        length = int(row[idx['Length']]) if idx['Length'] < len(row) else 0
    except (ValueError, IndexError):
        length = 0
    
    features['Length'] = length
    features['log_length'] = math.log(length) if length > 0 else 0
    
    # Interaction count (from 'Interacts with' column)
    interaction_text = row[idx['Interacts with']] if idx['Interacts with'] < len(row) else ''
    # Count interactors - rough estimate
    interaction_count = 0
    if interaction_text:
        # Count P: entries (UniProt IDs)
        interaction_count = len(re.findall(r'P:\d+', interaction_text))
        if interaction_count == 0:
            interaction_count = interaction_text.count(';') + 1 if ';' in interaction_text else 0
    
    features['interaction_count'] = interaction_count
    features['has_interactions'] = 1 if interaction_count > 0 else 0
    
    # Domain information (from 'Domain [CC]' column)
    domain_text = row[idx['Domain [CC]']] if idx['Domain [CC]'] < len(row) else ''
    domain_text = domain_text.upper()
    
    features['gpcr_domain'] = 1 if 'GPCR' in domain_text or 'G-PROTEIN COUPLED' in domain_text else 0
    features['ion_channel_domain'] = 1 if 'ION CHANNEL' in domain_text else 0
    features['transporter_domain'] = 1 if 'TRANSPORTER' in domain_text else 0
    features['receptor_domain'] = 1 if 'RECEPTOR' in domain_text else 0
    
    # Subcellular localization (from 'Subcellular location [CC]' column)
    location_text = row[idx['Subcellular location [CC]']] if idx['Subcellular location [CC]'] < len(row) else ''
    location_text = location_text.upper()
    
    features['plasma_membrane'] = 1 if 'PLASMA MEMBRANE' in location_text else 0
    features['nucleus'] = 1 if 'NUCLEUS' in location_text else 0
    features['cytoplasm'] = 1 if 'CYTOPLASM' in location_text else 0
    features['mitochondrion'] = 1 if 'MITOCHONDRION' in location_text else 0
    features['er'] = 1 if 'ENDOPLASMIC RETICULUM' in location_text else 0
    features['secreted'] = 1 if 'SECRETED' in location_text else 0
    
    # Calculate priors (weights based on biological relevance to TM helices)
    features['location_prior'] = (
        features['plasma_membrane'] * 0.5 +
        features['er'] * 0.3 +
        features['mitochondrion'] * 0.2
    )
    
    features['domain_prior'] = min(1.0, (
        features['gpcr_domain'] * 0.8 +
        features['ion_channel_domain'] * 0.7 +
        features['transporter_domain'] * 0.6 +
        features['receptor_domain'] * 0.4
    ))
    
    features['combined_prior'] = (features['location_prior'] + features['domain_prior']) / 2
    
    return features

def fetch_all_human_proteins():
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    fields = "accession,id,protein_name,gene_names,length,organism_name,cc_subcellular_location,cc_interaction,cc_disruption_phenotype,organism_id,cc_domain,ft_transmem"
    
    all_proteins = []  # Will store protein data
    headers = None
    start = 0
    size = 500  
    max_target = 200000  # Changed from 500,000 to 200,000
    tm_column_index = None
    total_fetched = 0
    total_skipped = 0
    
    print("=" * 60)
    print("FETCHING HUMAN PROTEINS WITH TRANSMEMBRANE HELICES")
    print("=" * 60)
    print(f"Target: {max_target:,} proteins with ≥1 TM helix\n")
    
    while len(all_proteins) < max_target:
        params = {
            "query": "organism_id:9606",  
            "format": "tsv",
            "fields": fields,
            "size": size,
            "start": start
        }
        
        print(f"Batch {start//size + 1}: Fetching entries {start} to {start + size}...")
        response = requests.get(base_url, params=params)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break
        
        lines = response.text.strip().split('\n')
        if not lines or len(lines) <= 1:
            print("No more data available from UniProt.")
            break
        
        if headers is None:
            headers = lines[0].split('\t')
            # Find the Transmembrane column
            try:
                tm_column_index = headers.index('Transmembrane')
                print(f"✓ Found 'Transmembrane' column at index {tm_column_index}")
                print(f"✓ Total columns: {len(headers)}")
                print(f"✓ Key columns: Entry, Length, Transmembrane, Subcellular location, Domain\n")
            except ValueError:
                print("Error: 'Transmembrane' column not found!")
                print(f"Available columns: {headers}")
                return None
        
        # Process each row
        page_rows = lines[1:] if start > 0 else lines[1:]
        proteins_in_batch = 0
        
        for row_str in page_rows:
            total_fetched += 1
            row = row_str.split('\t')
            
            # Check TM count from Transmembrane column
            if len(row) > tm_column_index:
                tm_value = row[tm_column_index].strip()
                tm_count = count_tm_helices(tm_value)
                
                # Only keep entries with ≥1 TM helix
                if tm_count > 0:
                    # Compute features for this protein
                    features = compute_features(row, headers)
                    
                    # Store the protein data
                    protein_data = {
                        'accession': row[headers.index('Entry')] if headers.index('Entry') < len(row) else '',
                        'id': row[headers.index('Entry Name')] if headers.index('Entry Name') < len(row) else '',
                        'gene_name': row[headers.index('Gene Names')] if headers.index('Gene Names') < len(row) else '',
                        'protein_name': row[headers.index('Protein names')] if headers.index('Protein names') < len(row) else '',
                        'length': int(row[headers.index('Length')]) if headers.index('Length') < len(row) else 0,
                        'tm_count': tm_count,
                        'features': features
                    }
                    
                    all_proteins.append(protein_data)
                    proteins_in_batch += 1
                    
                    if len(all_proteins) >= max_target:
                        break
                else:
                    total_skipped += 1
        
        print(f"  → Found {proteins_in_batch} proteins with TM helices")
        print(f"  → Total collected: {len(all_proteins):,}/{max_target:,}")
        print(f"  → Skipped (0 TM): {total_skipped:,}")
        if total_fetched > 0:
            print(f"  → Success rate: {len(all_proteins)/total_fetched*100:.1f}%\n")
        
        # Stop conditions
        if len(page_rows) < size:
            print("Reached end of UniProt data.")
            break
        
        if len(all_proteins) >= max_target:
            break
            
        start += size
        time.sleep(0.5)  # Be respectful to UniProt servers
    
    print("=" * 60)
    print(f"✓ COMPLETED! Collected {len(all_proteins):,} proteins with ≥1 TM helix")
    print(f"✓ Total fetched from UniProt: {total_fetched:,}")
    print(f"✓ Total skipped (0 TM): {total_skipped:,}")
    print("=" * 60)
    
    # Save the data
    if all_proteins:
        # Save main dataset with features
        output_file = f'uniprot_tm_positive_{len(all_proteins):,}_with_features.csv'
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            # Define all feature columns
            feature_columns = [
                'accession', 'id', 'gene_name', 'protein_name', 'length', 'tm_count',
                'Length', 'log_length', 'interaction_count', 'has_interactions',
                'gpcr_domain', 'ion_channel_domain', 'transporter_domain', 'receptor_domain',
                'plasma_membrane', 'nucleus', 'cytoplasm', 'mitochondrion', 'er', 'secreted',
                'location_prior', 'domain_prior', 'combined_prior'
            ]
            
            writer = csv.DictWriter(f, fieldnames=feature_columns)
            writer.writeheader()
            
            for protein in all_proteins:
                row_data = {
                    'accession': protein['accession'],
                    'id': protein['id'],
                    'gene_name': protein['gene_name'],
                    'protein_name': protein['protein_name'],
                    'length': protein['length'],
                    'tm_count': protein['tm_count'],
                    **protein['features']
                }
                writer.writerow(row_data)
        
        print(f"\n💾 Saved to: {output_file}")
        
        # Show statistics
        tm_counts = [p['tm_count'] for p in all_proteins]
        print(f"\n📊 Statistics:")
        print(f"  • Total proteins: {len(all_proteins):,}")
        print(f"  • Average TM count: {sum(tm_counts)/len(tm_counts):.2f}")
        print(f"  • Max TM count: {max(tm_counts)}")
        print(f"  • Min TM count: {min(tm_counts)}")
        
        # Distribution of TM counts
        tm_dist = Counter(tm_counts)
        print(f"\n📈 TM count distribution (top 10):")
        for tm_count in sorted(tm_dist.keys())[:10]:
            print(f"  • {tm_count} TM helix(s): {tm_dist[tm_count]:,} proteins ({tm_dist[tm_count]/len(all_proteins)*100:.1f}%)")
        
        # Show sample features
        print(f"\n🔬 Sample features (first protein):")
        sample = all_proteins[0]
        print(f"  Accession: {sample['accession']}")
        print(f"  Gene: {sample['gene_name']}")
        print(f"  TM count: {sample['tm_count']}")
        print(f"  Features:")
        for key, value in list(sample['features'].items())[:8]:
            print(f"    • {key}: {value}")
    
    return all_proteins

if __name__ == "__main__":
    fetch_all_human_proteins()