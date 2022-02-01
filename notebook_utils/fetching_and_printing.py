from typing import Callable
import numpy as np
import json
import os
from .filtering import filter_outliers, find_same_options
from .visualization import plot_Lipschitz_convergence, plot_Lipschitz_conv_iter


def print_M_and_SD(entries: list, keys: list=["Objective", "fairness error", "balance", "time"], printing: bool=True):
    """
    Return results for each key with mean and std, and print neatly.

    :param entries: entries from the results csv
    :param keys: keys/columns to report from the results csv
    :param printing: if set to False this function only returns results without printing

    :return dict of dicts like {"Objective": {"mean": 1, "std": 0.5}, ...}
    """
    results = {}
    for key in keys:
        sample = [float(entry[key]) for entry in entries]
        mean = np.mean(sample)
        std = np.std(sample)

        mean_str = f"M = {mean:.3f}"
        std_str = f"SD = {std:.3f}"
        if printing:
            print(f"      {key}{' '*(20-len(key))}{mean_str}{' '*(20-len(mean_str))}{std_str}{' '*(15-len(std_str))}({std/mean*100:.1f}%)")

        results[key] = {"mean": mean, "std": std}
    return results


def print_results(entries):
    """
    Print config info and then print mean and std for the keys in the unfiltered and then filtered entries.

    :param entries: entries from the results csv

    :return dict of dicts like {"Objective": {"mean": 1, "std": 0.5}, ...} from the filtered entries
    """
    entry = entries[0]
    print(f"    lmbda={entry['lmbda']}, Lipschitz={entry['L']}, runs={len(entries)}")
    
    print_M_and_SD(entries)

    filtered_entries, seeds = filter_outliers(entries, ["Objective", "fairness error", "balance"])
    printing = len(seeds) > 0
    if printing:
        print(f"\n      without {len(seeds)} outliers: {seeds}")
    return print_M_and_SD(filtered_entries, printing=printing)


def fetch_and_print(
configs: dict, configs_additional: dict, arg_getting_fn: Callable, MODES: list, DEFAULT_REPROD_L: float, CSV_NAME: str, FINAL_RESULTS_NAME: str) -> None:
    """
    Fetch results from the results csv, calculate means and stds, and neatly print them.

    :param configs: configurations to report, as defined in main.ipynb
    :param configs_additional: additional configurations to report, as defined in main.ipynb
    :param arg_getting_fn: function that gets the arguments for main() in test_fair_clustering.py, as defined in main.ipynb
    :param MODES: list of modes to report, as defined in main.ipynb
    :param DEFAULT_REPROD_L: default Lipschitz constant to use for reproduction
    :param CSV_NAME: the filename of the csv to read results from, eg "results.csv"
    :param FINAL_RESULTS_NAME: the filename to save the final results in, eg "final_results.json"
    """
    final_results = {}
    experiments_runtime = 0
    census_runtime = 0
    for mode in MODES:
        final_results[mode] = {}
        print(f"\n\n{mode.upper()}")
        for dataset in configs:
            print(f"\n  {dataset}")
            for cluster_option in configs[dataset]:
                if mode not in configs[dataset][cluster_option]:
                    continue

                print("\n    "+cluster_option.upper())

                lmbda = configs[dataset][cluster_option][mode]["lmbda"]
                L = DEFAULT_REPROD_L
                if "Lipschitz" in configs[dataset][cluster_option][mode]:
                    L = configs[dataset][cluster_option][mode]["Lipschitz"]

                args = arg_getting_fn(dataset=dataset, cluster_option=cluster_option, lmbda=lmbda, Lipschitz=L)
                existing_entries = find_same_options(CSV_NAME, args)
                
                if len(existing_entries) < 1:
                    print("no data yet")
                    continue

                runtime = sum([float(entry["time"]) for entry in existing_entries])
                experiments_runtime += runtime
                if dataset == "CensusII":
                    census_runtime += runtime
                results = print_results(existing_entries)

                if cluster_option not in final_results[mode]:
                    final_results[mode][cluster_option] = {}
                final_results[mode][cluster_option][dataset] = results
    with open(os.path.join(args.output_path, FINAL_RESULTS_NAME), "w") as f:
        json.dump(final_results, f, indent=4)

    additional_runtime = 0
    print(f"\n\n{'additional'.upper()}")
    for dataset in configs:
        print(f"\n  {dataset}")
        for cluster_option in configs[dataset]:
            use_configs = configs_additional[dataset][cluster_option]
            if len(use_configs) < 1:
                continue

            print("\n    "+cluster_option.upper())
            for config in use_configs:
                lmbda = config["lmbda"]
                L = DEFAULT_REPROD_L
                if "Lipschitz" in config:
                    L = config["Lipschitz"]

                args = arg_getting_fn(dataset=dataset, cluster_option=cluster_option, lmbda=lmbda, Lipschitz=L)
                existing_entries = find_same_options(CSV_NAME, args)
                
                if len(existing_entries) < 1:
                    print("no data yet")
                    continue

                runtime = sum([float(entry["time"]) for entry in existing_entries])
                additional_runtime += runtime
                if dataset == "CensusII":
                    census_runtime += runtime            
                print_results(existing_entries)

    print(f"Experiments runtime:    {experiments_runtime:.1f}    ({experiments_runtime/3600:.2f} Hours)")
    print(f"Census II runtime:      {census_runtime:.1f}    ({census_runtime/3600:.2f} Hours)")
    total_runtime = experiments_runtime + additional_runtime
    print(f"Total runtime:          {total_runtime:.1f}    ({total_runtime/3600:.2f} Hours)")


def fetch_and_print_Lipschitz(configs: dict, use_datasets: list, arg_getting_fn: Callable, LIPSCHITZ_CONSTANTS: list, CSV_NAME_LIPSCHITZ: str) -> None:
    """
    Fetch results from the results csv, calculate means and stds, and neatly print them.

    :param configs: configurations to report, as defined in main.ipynb
    :param use_datasets: list of datasets to use
    :param arg_getting_fn: function that gets the arguments for main() in test_fair_clustering.py, as defined in main.ipynb
    :param LIPSCHITZ_CONSTANTS: list of Lipschitz constants to report results for
    :param CSV_NAME_LIPSCHITZ: the filename of the csv to read results from, eg "results_Lipschitz.csv"
    """
    for dataset in use_datasets:
        if dataset == "CensusII":
            continue
        print("\n\n")
        for cluster_option in configs[dataset]:
            print(dataset+"\n  "+cluster_option.upper())
            energy_list_by_L = {}
            conv_iter_by_L = {}
            complete = True
            for L in LIPSCHITZ_CONSTANTS:
                args = arg_getting_fn(dataset=dataset, cluster_option=cluster_option, Lipschitz=L)
                existing_entries = find_same_options(CSV_NAME_LIPSCHITZ, args, keys=["dataset", "cluster_option", "L"])
                
                if len(existing_entries) < 1:
                    print(f"no data yet on {dataset} with {cluster_option} at Lipshitz={L}")
                    complete = False
                    continue

                keys = ["convergence_iter", "optimum", "time"]
                # filtered_entries = filter_outliers(existing_entries, keys)
                # print(f"\n    Lipschitz = {L}       (excluded {len(existing_entries)-len(filtered_entries)} outliers)")
                # existing_entries = filtered_entries

                energy_lists_by_run = []
                for entry in existing_entries:
                    with open(os.path.join(args.output_path, entry["energy_list_file"]), "r") as f:
                        energy_lists_by_run.append(json.loads(f.read()))

                max_len = max([int(entry["convergence_iter"]) for entry in existing_entries])
                energy_array_by_run = np.zeros((len(existing_entries), max_len))
                for i, energy_list in enumerate(energy_lists_by_run):
                    last_value = energy_list[-1]
                    for iter in range(len(energy_list), max_len):
                        energy_list.append(last_value)
                    energy_array_by_run[i] = energy_list
                    
                energy_list_by_L[L] = {
                    # "mean": np.mean(energy_array_by_run, axis=1),
                    "mean": energy_array_by_run[0],
                    "std": np.std(energy_array_by_run, axis=1),
                }

                # Change how optimum is displayed
                for i, entry in enumerate(existing_entries):
                    entry["optimum"] = energy_lists_by_run[i][-1]

                print(f"\n    Lipschitz = {L}")
                conv_iter_by_L[L] = print_M_and_SD(existing_entries, keys)["convergence_iter"]


            if complete:
                save_dir = os.path.join(args.output_path, dataset)
                save_path = os.path.join(save_dir, f"{cluster_option}_"+"Lipschitz_plot{suffix}.png")
                plot_Lipschitz_convergence(save_path, energy_list_by_L)
                # plot_Lipschitz_convergence(save_path, energy_list_by_L, yscale_log=True)

                save_path = os.path.join(save_dir, f"{cluster_option}_"+"conv-iter_Lipschitz_plot{suffix}.png")
                plot_Lipschitz_conv_iter(save_path, conv_iter_by_L)