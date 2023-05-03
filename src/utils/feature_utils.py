from Bio.SeqUtils.ProtParam import ProteinAnalysis

def extract_features(sequence):
    # Remove ambiguous amino acids (e.g., 'X') from the sequence
    clean_sequence = str(sequence).replace('X', '').replace('*', '')

    # Compute basic protein properties
    protein_analysis = ProteinAnalysis(clean_sequence)
    
    molecular_weight = protein_analysis.molecular_weight()
    isoelectric_point = protein_analysis.isoelectric_point()
    instability_index = protein_analysis.instability_index()
    aromaticity = protein_analysis.aromaticity()
    flexibility = protein_analysis.flexibility()
    gravy = protein_analysis.gravy()
    aliphatic_index = protein_analysis.aliphatic_index()
    boman_index = protein_analysis.boman_index()

    # Combine the features into a single array
    features = [
        molecular_weight,
        isoelectric_point,
        instability_index,
        aromaticity,
        flexibility,
        gravy,
        aliphatic_index,
        boman_index,
    ]
    amino_acid_percent = protein_analysis.get_amino_acids_percent()
    
    features.extend(amino_acid_percent.values())

    return features