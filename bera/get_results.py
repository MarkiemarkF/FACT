import json
import numpy as np
import os

def get_res(path="output/2022-01-19-11:01:40"):
    with open(path, 'r') as f:
        d = json.load(f)

    E = d['fair_score']

    label_unpr = d['assignment']
    N = len(d['points'])
    K = d['num_clusters']

    labels = []

    for i in range(N):
        for f in range(K):
            if label_unpr[i * K + f] == 1:
                labels.append(f)

    return E, labels


for dataset in ['student', 'german_credit', 'bank_red']:
    path = f'../bera_res/{dataset}.json'

    if not os.path.exists(path):
        E, l = get_res(f"output/{dataset}")
        res_dict = {}
        res_dict['E'] = E
        res_dict['l'] = l

        with open(path, 'w') as f:
            json.dump(res_dict, f)
