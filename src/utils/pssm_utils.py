import numpy as np
import json
from tqdm import tqdm


class PSSMUtils:

    @staticmethod
    def pssm_file_to_2Darray(pssm_path):
        with open(pssm_path, 'r') as file:
            lines = file.readlines()

            # Skip the header lines
            lines = lines[3:-6]

            # Extract the PSSM values
            pssm_values = []
            pssm_seq = []
            for line in lines:
                line_values = line.strip().split()[2:22]
                seq_values = line.strip().split()[1]
                pssm_values.append([int(value) for value in line_values])
                pssm_seq.append(seq_values)

            return "".join(pssm_seq), pssm_values
    
    @staticmethod
    def save_parsed_pssms(parsed_pssm):
        with open('../dbs/generated/parsed_pssm.json', 'w') as file:
            json.dump(parsed_pssm, file)

    @staticmethod
    def load_parsed_pssms_into_nparrays():
        with open('../dbs/generated/parsed_pssm.json', 'r') as file:
            parsed_pssm = json.load(file)
            print("Loaded PSSM data from JSON file.")

            # Convert the PSSM values to numpy arrays
            for key in tqdm(parsed_pssm):
                parsed_pssm[key] = np.array(parsed_pssm[key])

            print("Converted PSSM values to numpy arrays.")
            return parsed_pssm