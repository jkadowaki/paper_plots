#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

font = {'size': 18}

plt.rc('text', usetex=True)
plt.rc('font', **font)

bins = np.arange(0,1.01,0.1)

yagi_ar                = np.genfromtxt('yagi_axisratio.txt')
chen_ar                = np.genfromtxt('chen_axisratio.txt')
kadowaki_cluster_ar    = np.genfromtxt('kadowaki_cluster_axisratio.txt')
kadowaki_noncluster_ar = np.genfromtxt('kadowaki_non-cluster_axisratio.txt')
kadowaki_dense_ar      = np.genfromtxt('kadowaki_dense_axisratio.txt')
kadowaki_sparse_ar     = np.genfromtxt('kadowaki_sparse_axisratio.txt')

# HISTOGRAM
plt.figure(figsize=(8,6))
plt.hist(chen_ar,                bins=bins, histtype='stepfilled', color='blue',   density=True, linewidth=1, alpha=0.2, label=r"$\mathrm{Virgo \, Dwarfs \, (Chen \, et \, al. \, (2010))}$")
plt.hist(yagi_ar,                bins=bins, histtype='stepfilled', color='black',  density=True, linewidth=1, alpha=0.2, label=r"$\mathrm{Coma \, UDGs \, (Yagi \, et \, al. \, (2016))}$")
plt.hist(kadowaki_cluster_ar,    bins=bins, histtype='step',       color='green',  density=True, linewidth=3,            label=r"$\mathrm{Cluster \, UDGs \, (this \, work)}$")
plt.hist(kadowaki_noncluster_ar, bins=bins, histtype='step',       color='orange', density=True, linewidth=3,            label=r"$\mathrm{Non}$-$\mathrm{Cluster \, UDGs \, (this \, work)}$")
#plt.hist(kadowaki_dense_ar,      bins=bins, histtype='step', color='g', density=True, linewidth=2, label=r"$\mathrm{Kadowaki \, (Dense)}$")
#plt.hist(kadowaki_sparse_ar,     bins=bins, histtype='step', color='m', density=True, label=r"$\mathrm{Kadowaki \, (Sparse)}$")

plt.xlabel(r'$\mathrm{Axis \, Ratio}$', fontsize=22)
plt.legend(loc='upper left', fontsize=12, fancybox=True, shadow=True)
plt.savefig('axis_ratio.pdf', bbox_inches='tight')
plt.close()

bins = np.arange(0,1.01,0.01)

# CDF
plt.figure(figsize=(8,6))
plt.hist(chen_ar,                bins=bins, histtype='stepfilled', color='blue',   density=True, cumulative=True, linewidth=1, alpha=0.2, label=r"$\mathrm{Chen \, (Virgo \, Dwarfs)}$")
plt.hist(yagi_ar,                bins=bins, histtype='stepfilled', color='black',  density=True, cumulative=True, linewidth=1, alpha=0.2, label=r"$\mathrm{Yagi \, (Coma \, UDGs)}$")
plt.hist(kadowaki_cluster_ar,    bins=bins, histtype='step',       color='green',  density=True, cumulative=True, linewidth=3,            label=r"$\mathrm{Kadowaki \, (Cluster \, UDGs)}$")
plt.hist(kadowaki_noncluster_ar, bins=bins, histtype='step',       color='orange', density=True, cumulative=True, linewidth=3,            label=r"$\mathrm{Kadowaki \, (Non}$-$\mathrm{Cluster \, UDGs)}$")

plt.xlabel(r'$\mathrm{Axis \, Ratio}$', fontsize=22)
plt.ylabel(r'$\mathrm{CDF}$')
plt.legend(loc='upper left', fontsize=12, fancybox=True, shadow=True)
plt.savefig('ar_CDF.pdf', bbox_inches='tight')
plt.close()
