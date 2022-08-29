import json
import numpy as np
import matplotlib.pyplot as plt

# Sequences for which result is required to be compiled
sequences = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for i in sequences:

    seq = []

    with open(f"../json/{i}.json", "r") as f:
        for line in f :
            seq.append(json.loads(line))

        
    # Calculating total correctly classified static points
    TS = [val[str(list(val.keys())[0])]["TS"] for val in seq]

    # Calculating total static points present in ground truth
    total_static = [val[str(list(val.keys())[0])]["Total Voxels"] for val in seq]
    
    # Calculating total correctly classified dynamic points
    TD = [val[str(list(val.keys())[0])]["TD"] for val in seq]

    # Calculating total dynamic points present in ground truth
    total_dynamic = [val[str(list(val.keys())[0])]["Total Dynamic Points"] for val in seq]

    # Aggregatign the stats
    TS = np.sum(TS)
    total_static = np.sum(total_static)

    TD = np.sum(TD)
    total_dynamic = np.sum(total_dynamic)

    # Calculting accuracy and dynamic recall
    accuracy_our = ((TS/total_static) + (TD/(total_dynamic + 1e-8)))/2
    recall_our = TD/(total_dynamic + 1e-8)

    # Printing results
    print(i)
    print("accuracy ", accuracy_our)
    print("recall", recall_our)