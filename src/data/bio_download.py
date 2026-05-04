import requests
import csv
import time

def fetch_all_human_proteins():
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    fields = "accession,id,protein_name,gene_names,length,organism_name,cc_subcellular_location,cc_interaction,cc_disruption_phenotype,organism_id,cc_domain,ft_transmem"
    all_rows = []
    headers = None
    start = 0
    size = 500  
    max_entries = 60000  
    print("Fetching human proteins (including unreviewed entries)...")
    print(f"Will stop once {max_entries:,} entries are collected.\n")
    while True:
        params = {
            "query": "organism_id:9606",  
            "format": "tsv",
            "fields": fields,
            "size": size,
            "start": start
        }
        print(f"Fetching results {start} to {start + size}...")
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break
        lines = response.text.strip().split('\n')
        if not lines or len(lines) <= 1:
            break  
        if headers is None:
            headers = lines[0].split('\t')
        page_rows = lines[1:] if start > 0 else lines[1:]
        
        if len(all_rows) + len(page_rows) > max_entries:
            remaining = max_entries - len(all_rows)
            page_rows = page_rows[:remaining]
            all_rows.extend([row.split('\t') for row in page_rows])
            print(f"  Retrieved {len(page_rows)} entries. Total so far: {len(all_rows)}")
            print(f"\nReached {max_entries:,} entries. Stopping.")
            break
        else:
            all_rows.extend([row.split('\t') for row in page_rows])
            print(f"  Retrieved {len(page_rows)} entries. Total so far: {len(all_rows)}")
        if len(page_rows) < size:
            print("\nReached end of available data.")
            break
        start += size
        time.sleep(0.5)  
    print(f"\nDone! Total entries collected: {len(all_rows):,}")
    if headers and all_rows:
        output_file = f'uniprot_human_{len(all_rows):,}_entries.tsv'
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(headers)
            writer.writerows(all_rows)
        print(f"Saved to {output_file}")
    return headers, all_rows
if __name__ == "__main__":
    fetch_all_human_proteins()