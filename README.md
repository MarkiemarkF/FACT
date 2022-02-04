# Reproduction Variation Fair Clustering
<!-- This is the code for the AAAI 2021 paper: **[Variational Fair Clustering](https://arxiv.org/abs/1906.08207)**. This clustering method helps you to find clusters with specified proportions of different demographic groups pertaining to a sensitive attribute of the dataset (e.g. race, gender etc.), for any well-known clustering method such as K-means, K-median or Spectral clustering (Normalized cut) etc. in a flexible and scalable way. -->
This is the codebase for the Machine Learning Reproducibility Challenge (MLRC) of the paper **[Variational Fair Clustering](https://arxiv.org/abs/1906.08207)**.

## Requirements
1. Install Anaconda: https://www.anaconda.com/distribution/
2. The code is tested on Python 3.6 in . Refer to the [Getting started](#Getting-started) section for more detail.
3. Download the datasets [Bank](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing), [Adult](https://archive.ics.uci.edu/ml/datasets/adult), [Census II](https://archive.ics.uci.edu/ml/datasets/US+Census+Data+(1990)), [Student](https://archive.ics.uci.edu/ml/datasets/student+performance), and [Drugnet](https://sites.google.com/site/ucinetsoftware/datasets/covert-networks/drugnet), put the files in a directory with the name of the dataset with a capital in the [data/[dataset]](./data) directory.

## Getting started

Clone the repository

Create the environment necessary for running the experiments. Choose the command according to your operating system:
<!-- Then create and activate the environment necessary for running the experiments, using the following commands:-->
### Linux and MacOS
```bash
conda env create -f windows_macOS_fact_env.yaml
```
### Windows
```bash
conda env create -f linuxOS_fact_env.yaml
```
## Usage of environemnt
To activate the environment, use:
```bash
conda activate fact_vfc
```
To deactivate the environment, use:
```bash
conda deactivate
```

## Running the experiments
Our results can be displayed in main.ipynb by running the entire notebook without changing anything. To reproduce our results you can simply change the name of the "outputs" directory. This way a new directory by the name "outputs" will be created and filled with our results by running the entire notebook.

New experiments can also be conducted using the [test_fair_clustering.py](./test_fair_clustering.py) file. The usage of the file is specified as follows:
```bash
test_fair_clustering.py [--seed SEED] [-d DATASET]
                        [--cluster_option CLUSTER_OPTION]
                        [--kernel_type KERNEL_TYPE]
                        [--kernel_args KERNEL_ARGS]
                        [--lmbda LMBDA] [--lmbda-tune LMBDA-TUNE]
                        [--L L] [--data_dir DATA_DIR]
                        [--output_path OUTPUT_PATH]
                        [--plot_option_clusters_vs_lambda PLOT_OPTION_CLUSTERS_VS_LAMBDA]
                        [--plot_option_fairness_vs_clusterE PLOT_OPTION_FAIRNESS_VS_CLUSTERE]
                        [--plot_option_balance_vs_clusterE PLOT_OPTION_BALANCE_VS_CLUSTERE]
                        [--plot_option_convergence PLOT_OPTION_CONVERGENCE]
                        [--plot_bound_update PLOT_BOUND_UPDATE]
                        [--bera BERA]

optional arguments:
  --seed SEED       Fixed seed to initialise clusters
  -d DATASET        Name of the dataset to be used: Synthetic, Synthetic-unequal, Adult, Bank, CensusII
  --cluster_option CLUSTER_OPTION
                    Name of the cluster algorithm to be used: kmedian, kmean, ncut, kernel
  --kernel_type KERNEL_TYPE
                    Name of the kernel type to be used: poly, rad, tanh
  --kernel_args KERNEL_ARGS
                    Arguments to be used within the kernel function: x_y (where x and y are floats)
  --lmbda LMBDA     Initial lambda value
  --lmbda-tune LMBDA-TUNE
                    Whether lambda is tuned during clustering
  --L L             Lipschitz constant in the bound update
  --data_dir DATA_DIR
                    Datadirectory to retrieve datasets
  --output_path OUTPUT_PATH
                    Path where output values need to be stored
  --plot_option_clusters_vs_lambda PLOT_OPTION_CLUSTERS_VS_LAMBDA
                    Plot clusters in 2D w.r.t. lambda
  --plot_option_fairness_vs_clusterE PLOT_OPTION_FAIRNESS_VS_CLUSTERE
                    Plot clustering original energyt w.r.t. fairness
  --plot_option_balance_vs_clusterE PLOT_OPTION_BALANCE_VS_CLUSTERE
                    Plot clustering original energy w.r.t. balance
  --plot_option_convergence PLOT_OPTION_CONVERGENCE
                    Plot convergence of the fair clustering energy
  --plot_bound_update PLOT_BOUND_UPDATE
                    Plot (only one) boundy update
  --bera BERA       Whether Bera et al. results needed to be loaded and converted to metrics of Ziko et al.              
```

To view the notebook with our experimental results, run:
```bash
jupyter notebook main.ipynb
```

### Example run
```bash
$ python test_fair_clustering.py -d Synthetic --cluster_option kmedian --lmbda 10 --lmbda-tune False
```
