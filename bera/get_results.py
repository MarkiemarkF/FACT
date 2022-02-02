import os
import json


def get_res(path=os.path.join('output', '2022-01-19-11:01:40')):
    """
    Gets the Energy & labels from a bera fair clustering output file.

    :param path: path to dataset
    :return: clustering Energy & labels
    """
    # Open bera output
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


def convert_bera():
    """
    Converts results of bera et al. baseline to Ziko et al.

    :return: saves converted results as .json
    """
    # Get results from all files
    source_path = os.path.join("bera", "output")

    source_filepaths = [os.path.join(source_path, filename) for filename in os.listdir(source_path)]

    output_datasets = []
    for filepath in source_filepaths:
        dataset = os.path.basename(filepath)
        output_path = os.path.join('bera_res', dataset + '.json')

        E, l = get_res(filepath)
        res_dict = {}
        res_dict['dataset'] = dataset
        res_dict['E'] = E
        res_dict['l'] = l

        with open(output_path, 'w') as f:
            json.dump(res_dict, f, indent=4)

        output_datasets.append(dataset)
    return output_datasets


if __name__ == '__main__':
    convert_bera()
