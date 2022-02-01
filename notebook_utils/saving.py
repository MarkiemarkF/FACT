import os
import csv


def make_save_dict_Lipschitz(args, bound_energy_list: list=[0], elapsed: int=0, list_filepath: str="outputs/") -> dict:
    """
    Return dictionary to save as row in the results_Lipschitz.csv (where the results of the Lipschitz experiments go).

    :param args: Namespace object with arguments that were used for the run
    :param bound_energy_list: list of clustering energies (Objective) per iteration from the bound update
    :param elapsed: time taken for the bound update
    :param list_filepath: filepath to the logfile where the bound energy list is saved as json (because it's often too large for csv)

    :return dictionary with keys: "dataset", "lmbda", "cluster_option", "L", "convergence_iter", "optimum",
        "time", "seed", "energy_list_file"
    """
    return {
        "dataset": args.dataset,
        "lmbda": args.lmbda,
        "cluster_option": args.cluster_option,
        "L": args.L,                            # Lipschitz constant
        "convergence_iter": len(bound_energy_list),
        "optimum": min(bound_energy_list),
        "time": elapsed,                        # Time taken to finish this run
        "seed": args.seed,
        "energy_list_file": list_filepath,        
    }


def make_save_dict(args, results: dict={
    'N': 10000, 'J': 2, "clustering energy (Objective)": 200.00, "fairness error": 0.00, "balance": 1.00, "time": 100, 'K': 10,
}) -> dict:
    """
    Return dictionary to save as row in the results.csv

    :param args: Namespace object with arguments that were used for the run
    :param results: return value from main() in test_fair_clustering.py, consisting of:
        - 'N': Dataset size
        - 'J': Number of demographic groups (defined in dataset_load.py)
        - 'K': Number of clusters (defined in dataset_load.py)        
        - "clustering energy (Objective)": Discrete clustering energy
        - "fairness error"
        - "balance"
        - "time": Time taken to finish this run

    :return dictionary with keys: "dataset", "N", "J", "lmbda", "Objective", "fairness error", "balance", "cluster_option",
        "time", "seed", "lmbda_tune", "K", "L" 
    """
    return {
        "dataset": args.dataset,
        "N": results['N'],                                      # Dataset size
        "J": results['J'],                                      # Number of demographic groups (defined in dataset_load.py)
        "lmbda": args.lmbda,
        "Objective": results["clustering energy (Objective)"],  # Discrete clustering energy
        "fairness error": results["fairness error"],
        "balance": results["balance"],
        "cluster_option": args.cluster_option,
        "time": results["time"],                                # Time taken to finish this run
        "seed": args.seed,
        "lmbda_tune": args.lmbda_tune,                          
        "K": results['K'],                                      # Number of clusters (defined in dataset_load.py)
        "L": args.L,                                            # Lipschitz constant
    }    


def make_csv(dir_path: str, csv_path: str, fieldnames: list) -> None:
    """
    Make path and csv file with header if it doesn't exist yet. Otherwise do nothing.

    :param dir_path: full path to directory
    :param csv_path: full path to directory including the csv file
    :param fieldnames: list of fieldnames for the csv
    """
    os.makedirs(dir_path, exist_ok=True)
    if os.path.isfile(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            if sum(1 for row in reader) > 0:
                return

    with open(csv_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames)
        writer.writeheader()