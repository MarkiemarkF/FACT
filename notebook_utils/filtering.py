import csv
import os
from cv2 import threshold
import numpy as np
from scipy.stats import norm


def compare_entry(args, entry: dict, keys: list) -> bool:
    """
    Compare args to an entry from the csv file and return True if they have the given keys in common:

    :param args: arguments that will be passed to main() in test_fair_clustering.py
    :param entry: entry in the csv file made by run_main(), as defined in main.ipynb
    :param keys: list of keys to compare

    :return True if the values of the keys are the same
    """
    for key in keys:
        if str(getattr(args, key)) != entry[key]:
            return False
    return True


def find_same_options(csv_name: str, args, keys: list=["dataset", "lmbda", "cluster_option", "lmbda_tune", "L"]) -> list:
    """
    Return entries from the csv file that have the same configs as the passed args, using compare_entry().

    :param csv_name: filename of the results csv, eg "results.csv"
    :param args: arguments that will be passed to main() in test_fair_clustering.py
    :param keys: list of keys to compare

    :return list of entries (dicts)
    """
    entries = []
    csv_path = os.path.join(args.output_path, csv_name)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if compare_entry(args, row, keys):
                entries.append(row)
    return entries


def filter_outliers(entries: list, keys: list, threshold_fac: float=4) -> tuple:
    """
    Use Chauvenet outlier detection to filter outliers.

    :param entries: list of entries from the csv file made by run_main(), as defined in main.ipynb
    :param keys: keys of the entries to detect outliers by
    :param threshold_fac: determines the threshold for outlier detection, lower means less sensitive detection

    :return tuple: (
        list of filtered entries,
        the seeds of the outliers,
    )
    """
    outlier_seeds = set()

    N = len(entries)
    T_N = abs(norm.ppf(1/(threshold_fac*N)))
    for key in keys:
        sample = [float(entry[key]) for entry in entries]
        mean = np.mean(sample)
        std = np.std(sample)
        for entry in entries:
            distance = abs(float(entry[key]) - mean) / (std + 0.01*mean)
            if distance > T_N:
                outlier_seeds.add(entry["seed"])
    new_entries = []
    for entry in entries:
        if entry["seed"] in outlier_seeds:
            continue
        new_entries.append(entry)
    return new_entries, outlier_seeds