# Data Dictionary

## Target Variable: `tm_helix_count`
- Number of transmembrane helices in the protein
- Type: integer count (0-12 typically)

## Features

| Feature | Type | Description |
|---------|------|-------------|
| Length | numerical | Protein sequence length (number of amino acids) |
| log_length | numerical | Log-transformed length (handles skew) |
| interaction_count | numerical | Number of known protein interaction partners |
| location_prior | numerical | Prior probability of TM helices based on subcellular localization |
| domain_prior | numerical | Prior expectation of TM count based on domain family |
| combined_prior | numerical | Weighted combination of location and domain priors |
| gpcr_domain | binary | Protein contains GPCR domain (binary) |
| ion_channel_domain | binary | Protein contains ion channel domain (binary) |
| transporter_domain | binary | Protein contains transporter domain (binary) |
| receptor_domain | binary | Protein contains receptor domain (binary) |
| plasma_membrane | binary | Protein localizes to plasma membrane (binary) |
| nucleus | binary | Protein localizes to nucleus (binary) |
| cytoplasm | binary | Protein localizes to cytoplasm (binary) |
| mitochondrion | binary | Protein localizes to mitochondrion (binary) |
| er | binary | Protein localizes to endoplasmic reticulum (binary) |
| secreted | binary | Protein is secreted (binary) |
| has_interactions | binary | Protein has known interaction partners (binary) |
