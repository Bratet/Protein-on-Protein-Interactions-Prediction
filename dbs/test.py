import json
FEATURES_PATH = r"C:\Users\AHMED MRABET\Documents\ENSIAS-Workspace\PFA\Protein-on-Protein-Interactions-Prediction\dbs\covid\id_features_data2.json"

with open(FEATURES_PATH, 'r') as f:
    data = json.load(f)
    
sequences_json = data.keys()


import pandas as pd

interactions = pd.read_csv(r"C:\Users\AHMED MRABET\Documents\ENSIAS-Workspace\PFA\Protein-on-Protein-Interactions-Prediction\dbs\covid\ultimate_data.csv")

# print(len(interactions[interactions['Interactor A'].isin(sequences_json)]))
# print(len(interactions[interactions['Interactor B'].isin(sequences_json)]))
# # remove rows where the protein is not in the dataset
# interactions = interactions[interactions['Interactor A'].isin(sequences_json)]
# interactions = interactions[interactions['Interactor B'].isin(sequences_json)]

# # save the new dataset
# interactions.to_csv(r"C:\Users\AHMED MRABET\Documents\ENSIAS-Workspace\PFA\Protein-on-Protein-Interactions-Prediction\dbs\covid\data2_no_self_interactions.csv", index=False)


import numpy as np
# count number of proteins that have nan in their vectors of features
proteins_with_nan = set()
count = 0
for key in data.keys():
    # get features of the protein
    features = data[key]
    for feature in features.values():
        for num in feature:
            if np.isnan(num):
                count += 1
                proteins_with_nan.add(key)
                
            
            
            
print(count)
proteins_with_nan = list(proteins_with_nan)
print(len(proteins_with_nan))

# remove these proteins from json file
for protein in proteins_with_nan:
    del data[protein]
    
# save the new json file
with open(r"C:\Users\AHMED MRABET\Documents\ENSIAS-Workspace\PFA\Protein-on-Protein-Interactions-Prediction\dbs\covid\id_features_data2.json", 'w') as f:
    json.dump(data, f)