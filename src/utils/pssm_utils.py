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
            
            # print(f"{''.join(pssm_seq)} | {pssm_values}")
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
    
    @staticmethod
    def update_parsed_pssms(new_pssms):
        with open('../dbs/generated/parsed_pssm.json', 'r') as file:

            parsed_pssm = json.load(file)
            print("Loaded PSSM data from JSON file.")

            # save the number of pssms before updating
            num_pssms_before = len(parsed_pssm)

            # add new pssms to the parsed_pssm dictionary
            parsed_pssm.update(new_pssms)

            # save the number of pssms after updating
            num_pssms_after = len(parsed_pssm)

            # printing the number of pssms before and after updating
            print(f"Number of PSSMs added: {num_pssms_after - num_pssms_before}")
        
        # save the updated parsed_pssm dictionary to the json file
        with open('../dbs/generated/parsed_pssm.json', 'w') as file:
            json.dump(parsed_pssm, file)
            print("Updated PSSM data saved to JSON file.")
    
    @staticmethod
    def pssm_to_postgresql(lst):
        """
        Convert a 2D Python list of integers to a PostgreSQL 2D array string.
        
        Args:
        lst (list): A 2D Python list of integers.
        
        Returns:
        str: A PostgreSQL 2D array string representation of the input list.
        """
        if len(lst) == 2 and type(lst[1] == int):
            lst = lst[0]
        
        result = '{'
        for i, row in enumerate(lst):
            result += '{'
            for j, value in enumerate(row):
                result += str(value)
                if j < len(row) - 1:
                    result += ','
            result += '}'
            if i < len(lst) - 1:
                result += ','
        result += '}'
        return result
        