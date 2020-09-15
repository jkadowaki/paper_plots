#!/usr/bin/env python

from scipy import stats
import numpy as np

yagi       = np.genfromtxt('yagi_axisratio.txt')
sparse     = np.genfromtxt('kadowaki_sparse_axisratio.txt')
dense      = np.genfromtxt('kadowaki_dense_axisratio.txt')
cluster    = np.genfromtxt('kadowaki_cluster_axisratio.txt')
noncluster = np.genfromtxt('kadowaki_non-cluster_axisratio.txt')

print("Mean Values:")
print("<Sparse>      =", np.mean(sparse))
print("<Dense>       =", np.mean(dense))
print("<Cluster>     =", np.mean(cluster))
print("<Non-Cluster> =", np.mean(noncluster))
print("<Yagi>        =", np.mean(yagi))

print("\nLocal Env: Sparse vs. Dense")
print(stats.ks_2samp(sparse, dense))
print(stats.ttest_ind(sparse, dense, equal_var=False))
print(stats.anderson_ksamp([sparse, dense]))

print("\n\nGlobal Env: Cluster vs. Non-Cluster")
print(stats.ks_2samp(noncluster, cluster))
print(stats.ttest_ind(noncluster, cluster, equal_var=False))
print(stats.anderson_ksamp([noncluster, cluster]))

print("\n\nKadowaki (non-cluster) vs. Yagi (cluster)")
print(stats.ks_2samp(yagi, noncluster))
print(stats.ttest_ind(yagi, noncluster, equal_var=False))
print(stats.anderson_ksamp([yagi, noncluster]))
print("\n")
