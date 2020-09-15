#!/usr/bin/env python

################################################################################

import sys
sys.path.append("../code")

from read_data import read_data
import numpy as np
from numpy.random import randint, seed

import matplotlib.pyplot as plt
font = {'size': 18}
plt.rc('text', usetex=True)
plt.rc('font', **font)

from scipy.stats import ttest_ind_from_stats, ttest_ind

# Seed the Random Number Generator for Reproducible Results
#seed(0)

# Constants
COMA_VELOCITY = 6925

################################################################################

def compute_RMSE(arr, val=None):
    if val==None:
        val = np.mean(arr)

    return np.sqrt(np.mean((val - arr)**2))


################################################################################

def bootstrap_errors(arr, realizations=1000, plot_fname=None, verbose=False):
    """
    Compute errors of a statistical parameter measured in a distribution.
    
    Args:
        arr (np.ndarray): Distribution.
        realizations (int): Number of realizations in the simulation.
        plot_fname (str): Plot distribution. If set to None, plot is not generated.
    """

    means = np.empty(realizations)
    rmses = np.empty(realizations)

    for count in range(realizations):

        # List of indices for realization
        indices = randint(low=0, high=len(arr)-1, size=len(arr))
        values  = [arr[idx] for idx in indices]

        # Compute parameter
        means[count] = np.mean(values)
        rmses[count] = compute_RMSE(np.array(values), val=COMA_VELOCITY)


    print("Sample Size:   ", len(arr))
    print("# Realizations:",   realizations)

    print("\nDistribution Mean: ", np.mean(arr))
    print("Distribution StDev:",   np.std(arr))
    print("Distribution RMSE: ",   compute_RMSE(arr, val=COMA_VELOCITY))

    if verbose:
        print("MEAN \tRMSEs")
        for (avg,rmse) in zip(means, rmses):
            print(avg, "\t", rmse)

    print("\nMean of Sampling Means:    ", np.mean(means))
    print(  "Standard Error of the Mean:", np.std(means))
    print(  "Mean of Sampling RMSEs:    ", np.mean(rmses))
    print(  "Standard Error of RMSEs:   ", np.std(rmses))


    # Create Figure
    ax1_bins = 10
    ax2_bins = 100
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8,15))

    # Original Distribution
    mean1 = np.mean(arr)
    rmse1 = compute_RMSE(arr, val=COMA_VELOCITY)
    ax1.hist(arr, ax1_bins, density=False, color="b")
    y1min, y1max = ax1.get_ylim()
    ax1.plot([COMA_VELOCITY, COMA_VELOCITY], [y1min,y1max], linestyle='-',  lw=1, c="black",  
             label=r"$\overline{cz}_\mathrm{coma} \, \mathrm{(km \, s}^{-1})$")
    ax1.plot([mean1,     mean1],             [y1min,y1max], linestyle='-',  lw=3, c="red",
             label=r"$\mu \, (\mathrm{km \, s}^{-1})$")
    ax1.plot([mean1+rmse1,mean1+rmse1],        [y1min,y1max], linestyle='--', lw=1, c="orange",
             label=r"$\mu + \mathrm{RMSE}$")
    ax1.plot([mean1-rmse1,mean1-rmse1],        [y1min,y1max], linestyle='--', lw=1, c="orange",
             label=r"$\mu - \mathrm{RMSE}$")
    ax1.set_ylim(y1min, y1max)
    ax1.legend(loc='upper left', fontsize=12)

    # Bootstrapped Means
    mean2 = np.mean(means)
    std2  = np.std(means)
    ax2.hist(means, ax2_bins, density=False, color="b")
    y2min, y2max = ax2.get_ylim()
    ax2.plot([COMA_VELOCITY, COMA_VELOCITY], [y2min,y2max], linestyle='-',  lw=1, c="black",
             label=r"$\overline{cz}_\mathrm{coma} \, \mathrm{(km \, s}^{-1})$")
    ax2.plot([mean2,     mean2],             [y2min,y2max], linestyle='-',  lw=3, c="red",
             label=r"$\overline{x} \, (\mathrm{km \, s}^{-1})$")
    ax2.plot([mean2+std2,mean2+std2],        [y2min,y2max], linestyle='--', lw=1, c="orange",
             label=r"$\overline{x} + 1s_x$")
    ax2.plot([mean2-std2,mean2-std2],        [y2min,y2max], linestyle='--', lw=1, c="orange",
             label=r"$\overline{x} - 1s_x$")
    ax2.set_ylim(y2min, y2max)
    ax2.legend(loc='upper left', fontsize=12)

    ax2.text(0.67,0.90, r"$\mathrm{Sample \, Size:}$"     + "\t" + r"${}$".format(len(arr)),
             transform=ax2.transAxes, fontsize=14)
    ax2.text(0.67,0.83, r"$\mathrm{\# \, Realizations:}$" + "\t" + r"${}$".format(realizations),
             transform=ax2.transAxes, fontsize=14)

    # Bootstrapped RMSEs
    mean3 = np.mean(rmses)
    std3  = np.std(rmses)
    ax3.hist(rmses, ax2_bins, density=False, color="b")
    y3min, y3max = ax3.get_ylim()
    ax3.plot([mean3,     mean3],             [y3min,y3max], linestyle='-',  lw=3, c="red",
             label=r"$\overline{\mathrm{RMSE}} \, (\mathrm{km \, s}^{-1})$")
    ax3.plot([mean3+std3,mean3+std3],        [y3min,y3max], linestyle='--', lw=1, c="orange",
             label=r"$\overline{\mathrm{RMSE}} + 1s_\mathrm{RMSE}$")
    ax3.plot([mean3-std3,mean3-std3],        [y3min,y3max], linestyle='--', lw=1, c="orange",
             label=r"$\overline{\mathrm{RMSE}} - 1s_\mathrm{RMSE}$")
    ax3.set_ylim(y3min, y3max)
    ax3.legend(loc='upper left', fontsize=12)

    ax3.text(0.67,0.90, r"$\mathrm{Sample \, Size:}$"     + "\t" + r"${}$".format(len(arr)),
             transform=ax3.transAxes, fontsize=14)
    ax3.text(0.67,0.83, r"$\mathrm{\# \, Realizations:}$" + "\t" + r"${}$".format(realizations),
             transform=ax3.transAxes, fontsize=14)


    # Set Axis Labels
    ax1.set_xlabel(r"$\mathrm{Original \,Velocity \, Distribution \, (km \, s^{-1})}$", fontsize=18)
    ax1.set_ylabel(r"$\mathrm{Frequency}$", fontsize=18)
    ax2.set_xlabel(r"$\mathrm{Distribution \, of \, the \, Mean \, Velocity \, (km \, s^{-1})}$", fontsize=18)
    ax2.set_ylabel(r"$\mathrm{Frequency}$", fontsize=18)
    ax3.set_xlabel(r"$\mathrm{Distribution \, of \, the \, RMSE \, (km \, s^{-1})}$", fontsize=18)
    ax3.set_ylabel(r"$\mathrm{Frequency}$", fontsize=18)
       
 
    # Save Figure
    plt.savefig(plot_fname, bbox_inches='tight')
    plt.close()


    return mean2, std2, mean3, std3, means, rmses

################################################################################


if __name__ == "__main__":

    num_realizations = 50000

    df        = read_data("../data/kadowaki2019.tsv")
    df_dense  = df.loc[(df["LocalEnv"]=="Dense")  & (df["GlobalEnv"]=="Cluster")]
    df_sparse = df.loc[(df["LocalEnv"]=="Sparse") & (df["GlobalEnv"]=="Cluster")]

    print("\n\n----------------------------------------------")
    print(  "\n~~~~~~~~~~~~~~~~~~~~DENSE~~~~~~~~~~~~~~~~~~~~~")
    meanD, meanD_err, rmseD, rmseD_err, meanD_list, rmseD_list = bootstrap_errors(df_dense["cz"].values,  realizations=num_realizations, plot_fname="../plots/bootstrap_dense.pdf")
    print(  "\n~~~~~~~~~~~~~~~~~~~~SPARSE~~~~~~~~~~~~~~~~~~~~")
    meanS, meanS_err, rmseS, rmseS_err, meanS_list, rmseS_list = bootstrap_errors(df_sparse["cz"].values, realizations=num_realizations, plot_fname="../plots/bootstrap_sparse.pdf")

    # Hypothesis testing
    print("\n----------------------------------------------")
    print("\n~~~~~~~~~~Hypothesis Testing (MEANs)~~~~~~~~~~")
    print(ttest_ind_from_stats(mean1=meanD, std1=meanD_err, nobs1=num_realizations,
                               mean2=meanS, std2=meanS_err, nobs2=num_realizations,
                               equal_var=False))
    print("\n~~~~~~~~~~Hypothesis Testing (RMSEs)~~~~~~~~~~")
    print(ttest_ind_from_stats(mean1=rmseD, std1=rmseD_err, nobs1=num_realizations,
                               mean2=rmseS, std2=rmseS_err, nobs2=num_realizations,
                               equal_var=False))
    print("\n----------------------------------------------\n")
    
    
    
        
    # Hypothesis testing
    print("\n----------------------------------------------")
    print("\n~~~~~~~~~~Hypothesis Testing (MEANs)~~~~~~~~~~")
    dense_rmse  = compute_RMSE(df_dense["cz"].values,  val=COMA_VELOCITY)
    sparse_rmse = compute_RMSE(df_sparse["cz"].values, val=COMA_VELOCITY)

    print(ttest_ind(dense_rmse, sparse_rmse, equal_var=False))
