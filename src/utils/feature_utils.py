from Bio.SeqUtils.ProtParam import ProteinAnalysis

def extract_features(sequence):
    # Remove ambiguous amino acids (e.g., 'X') from the sequence
    clean_sequence = str(sequence).replace('X', '').replace('*', '')

    # Compute basic protein properties
    protein_analysis = ProteinAnalysis(clean_sequence)
    molecular_weight = protein_analysis.molecular_weight()
    isoelectric_point = protein_analysis.isoelectric_point()
    amino_acid_percent = protein_analysis.get_amino_acids_percent()

    # Combine the features into a single array
    features = [
        molecular_weight,
        isoelectric_point,
    ]
    features.extend(amino_acid_percent.values())

    return features