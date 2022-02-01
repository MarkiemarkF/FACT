from matplotlib import pyplot as plt


def plot_Lipschitz_convergence(save_path, energy_list_by_L):
    """
    Plot clustering objective by iteration for different Lipschitz constants

    :param save_path: path to save the plot in
    :param energy_list_by_L: dict (by Lipschitz value) of listed means and stds of bound update energies like:
        {
            0.1: [100, 90, 70, 50]
        }
    """
    fig, ax = plt.subplots(1,1,figsize=(5,4))
    for L, lis in energy_list_by_L.items():
        ax.plot(range(len(lis)), lis, label=f"L = {L}")
    ax.legend()
    ax.set_ylabel("fair objective")
    ax.set_xlabel("iterations")
    suffix = ""
    margin = 0.05

    max_val = max(energy_list_by_L[0.01][1:])
    min_val = energy_list_by_L[0.01][-1]
    y_range = max_val - min_val
    ax.set_ylim(min_val-y_range*margin, max_val+y_range*margin)

    x_range = max([len(lis) for lis in energy_list_by_L.values()])
    if x_range > 100:
        ax.set_xscale('log')
    plt.savefig(save_path.format(suffix=suffix))
    plt.show()
    plt.close('all')


def plot_Lipschitz_conv_iter(save_path, conv_iter_by_L):
    """
    Plot the iterations it took to converge for different Lipschitz constants as a bar chart.

    :param save_path: path to save the plot in
    :param conv_iter_by_L: dict (by Lipschitz value) of mean and stds of bound update convergence iterations like:
        {
            0.1: {"mean": 100, "std": 10},
            0.01: {"mean": 90, "std": 10},
            0.001: {"mean": 50, "std": 10},
        }
    """   
    fig, ax = plt.subplots(1,1,figsize=(5,4))
    for L, conv_iter in conv_iter_by_L.items():
        ax.bar(str(L), conv_iter["mean"], color="darkred")
    ax.set_ylabel("iterations to convergence in bound update")
    ax.set_xlabel("Lipschitz constant")
    if max([conv_iter["mean"] for conv_iter in conv_iter_by_L.values()]) > 500:
        ax.set_yscale('log')
    plt.savefig(save_path.format(suffix=""))
    plt.show()
    plt.close('all')