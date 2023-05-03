from Bio.SeqUtils.ProtParam import ProteinAnalysis

def aliphatic_index(sequence):
    aliphatic_amino_acids = {'A': 89.09, 'V': 174.09, 'I': 166.15, 'L': 166.15}
    total_weight = sum(aliphatic_amino_acids.get(aa, 0) for aa in sequence)
    return total_weight / len(sequence)

def extract_features(sequence):
    # Remove ambiguous amino acids (e.g., 'X') and other unexpected codes from the sequence
    clean_sequence = ''.join(aa for aa in str(sequence) if aa in 'ACDEFGHIKLMNPQRSTVWY')

    # Compute basic protein properties
    protein_analysis = ProteinAnalysis(clean_sequence)
    
    molecular_weight = protein_analysis.molecular_weight()
    isoelectric_point = protein_analysis.isoelectric_point()
    instability_index = protein_analysis.instability_index()
    aromaticity = protein_analysis.aromaticity()
    gravy = protein_analysis.gravy()
    
    # Compute secondary structure content
    secondary_structure_fraction = protein_analysis.secondary_structure_fraction()
    helix, sheet, coil = secondary_structure_fraction

    # Compute aliphatic index
    aliphatic_index_value = aliphatic_index(clean_sequence)

    # Compute net charge at pH 7
    charged_aa_content = protein_analysis.charge_at_pH(7)
    
    # Compute cysteine content
    cysteine_content = clean_sequence.count('C') / len(clean_sequence)
    
    # Combine the features into a single array
    features = [
        molecular_weight,
        isoelectric_point,
        instability_index,
        aromaticity,
        gravy,
        helix,
        sheet,
        coil,
        aliphatic_index_value,
        charged_aa_content,
        cysteine_content
    ]
    
    amino_acid_percent = protein_analysis.get_amino_acids_percent()
    
    features.extend(amino_acid_percent.values())

    return features
