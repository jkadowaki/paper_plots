#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
#plt.rcParams.update({'font.size':12,'text.usetex':True})


##################################### DATA #####################################


tasks  = ["QQP", "ARC", "WikiQA", "NER"]
labels = ["P_c", "O_c", "C_c", "POC_c", "P_f", "O_f", "C_f", "POC_f", ]
colors = ['r', 'g', 'b', 'k']

seed_ceil = np.array([ [0.21,   2.52,   0.08,   0.73],
                       [1.03,   3.81,   3.30,   1.22],
                       [3.45,   15.18,  1.98,   2.85],
                       [0.004,  0.019,  0.020,  0.003] ])

seed_floor = np.array([ [ 1.83,   0.06,   2.51,   0.81],
                        [ 4.70,   1.73,   1.41,   3.50],
                        [ 11.93,  1.12,   12.33,  10.20],
                        [ 0.03,   0.01,   0.013,  0.04] ])

epoch_ceil = np.array([ [0.13,  0.43,   0.01,   0.04],
                        [0.003, 2.99,   2.99,   2.99],
                        [0.32,  0.27,   0.04,   0.27],
                        [0.002, 0.009,  0.01,   0.002] ])
    
epoch_floor = np.array([ [0.74,   0.29,   1.26,   0.96],
                         [2.59,   0.30,   0.30,   0.30],
                         [0.43,   0.79,   1.022,  0.79],
                         [0.02,   0.005,  0.007,  0.02] ])


nrow, ncol = np.shape(seed_ceil)
xceil  = range(ncol)
xfloor = range(ncol, 2*ncol)


plt.figure()

for idx in range(nrow):
    plt.plot(xceil,  seed_ceil[idx],   colors[idx], linestyle='-',  label=tasks[idx]+' seed')
    plt.plot(xceil,  epoch_ceil[idx],  colors[idx], linestyle='--',  label=tasks[idx]+' epoch')
    plt.plot(xfloor, seed_floor[idx],  colors[idx], linestyle='-')
    plt.plot(xfloor, epoch_floor[idx], colors[idx], linestyle='--')


plt.xlabel("Ensemble Indicator")
plt.ylabel("MSE from Ceiling/Floor Ensemble")

plt.yscale('log')
plt.xticks(range(2*ncol),  labels)
plt.legend(bbox_to_anchor=(0.95, 0.5))

plt.savefig("mse_plot.pdf", bbox_inches='tight')
plt.close()
