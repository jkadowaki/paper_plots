#!/usr/bin/env python

from __future__ import print_function
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 16}
plt.rc('font', **font)

import numpy as np
import os
import pandas as pd
from scipy import stats
from sklearn import linear_model
import seaborn as sns
import sys

sys.path.append('../code')
from pair_plot import read_data

# CONSTANTS
h0 = 0.6737


################################################################################

def schechter_lf(band="r", h=h0):
    """
    Returns the Canonical Cluster Galaxy Luminosity L*
    """
    if band=="B":
        return -20.6
    elif band=="r":
        return -20.73 + 5*np.log10(h)
    else:
        raise Exception("Unknown band!")

################################################################################

def mag_multiple(mag, lum_frac):
    return mag -2.5 * np.log10(lum_frac)

def lum_fraction(mag1, mag2):
    return 10 ** (-(mag1 - mag2)/2.5)

################################################################################

def main(data_name='kadowaki2019.tsv',
         plot_name='red_sequence.pdf',
         data_directory='../data',
         plot_directory='../plots',
         udg_only=True,
         local_env=True,
         density=False):
    
    """
    Args:
        (str)  data_file:
        (str)  plot_file:
        (str)  data_directory:
        (str)  plot_directory:
        (bool) udg_only:
    """

    # Data File
    data_file = os.path.join(data_directory, data_name)

    # Save to Plots
    plot_file = os.path.join(plot_directory, plot_name)
    
    # Load Data
    df_results = read_data(data_file, udg_only=udg_only, field='Coma')
    efeat      = 'Density' if density else ('LocalEnv' if local_env else 'GlobalEnv')
    df_results = df_results.sort_values(by=[efeat])
    df_results = df_results.reset_index(drop=True)
    
    
    # Mean Elliptical Colors -- Schombert (2016)
    L_mag       = schechter_lf(band="r", h=h0)
    lum_bins    = np.array([   2.,    1.,   0.5,   0.2,   0.1])
    mag_bin     = mag_multiple(L_mag, lum_bins)

    gr_color    = np.array([0.829, 0.816, 0.801, 0.775, 0.741])
    gr_sigma    = np.array([0.024, 0.021, 0.030, 0.028, 0.047])
    limit_color = gr_color - 2*gr_sigma
    
    
    # Plot
    xfeat   = "Mr"
    yfeat   = "g-r"
    xrange  = np.array([-22.5, -13.5])
    yrange  = np.array([0.1, 0.9])
    xrange2 = [lum_fraction(mag, L_mag) for mag in xrange]
    
    
    # Extrapolate Colors
    gr_lm        = linear_model.LinearRegression()
    gr_model     = gr_lm.fit(mag_bin.reshape(-1,1), gr_color)
    gr_bounds    = gr_model.predict(xrange.reshape(-1,1))
    limit_lm     = linear_model.LinearRegression()
    limit_model  = limit_lm.fit(mag_bin.reshape(-1,1), limit_color)
    limit_bounds = limit_model.predict(xrange.reshape(-1,1))
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
    for idx,feat in enumerate(['LocalEnv', 'GlobalEnv', 'Density']):
                             
        # Plot Red Sequence Extrapolation
        ax[idx].plot(xrange, gr_bounds,    c='r', lw=1, ls='--')
        ax[idx].plot(xrange, limit_bounds, c='b', lw=1, ls='--')
        ax[idx].fill_between(xrange, limit_bounds, 999,
                             facecolor='red', alpha=0.25)
        
        # Plot Red Sequence
        ax2 = ax[idx].twiny() # Creates New X-Axis that Shares Y-Axis
        ax[idx].plot(mag_bin, gr_color,  c='r', lw=3)
        (_, caps, _) = ax[idx].errorbar(mag_bin, gr_color, yerr=2*gr_sigma,
                                        fmt="o", c='r', markeredgecolor='k',
                                        ecolor='b', elinewidth=2,
                                        capsize=3, capthick=3, uplims=True)
        for cap in caps:
            cap.set_color('b')
            cap.set_marker('_')
        #ax[idx].plot(mag_bin, limit_color, c='b', lw=3)
        ax[idx].fill_between(mag_bin, limit_color, 999,
                             facecolor='red', alpha=0.25)
        
        # Plot UDGs
        values    = np.unique(df_results[feat])
        df_dense  = df_results[df_results[feat] == values[0]]
        df_sparse = df_results[df_results[feat] == values[1]]
        ax[idx].scatter(df_dense[xfeat],  df_dense[yfeat],  label=values[0],
                        marker='^', c='Green',  edgecolors='k', linewidth=0.5)
        ax[idx].scatter(df_sparse[xfeat], df_sparse[yfeat], label=values[1],
                        marker='o', c='Orange', edgecolors='k', linewidth=0.5)
        # Subplot Legends & Formatting
        title = r"$\mathrm{Local \, Environment}$"  if feat=="LocalEnv"  else \
                r"$\mathrm{Global \, Environment}$" if feat=="GlobalEnv" else \
                r"$\mathrm{Environment \, Density}$"
        ax[idx].set_xlim(xrange)
        ax[idx].set_xlabel(r"$M_r$")
        ax2.set_xlim(xrange2)
        ax2.set_xscale('log')
        ax2.set_xlabel(r"$L_r / L_r^*$")
        ax[idx].set_ylim(yrange)
        if idx==0:
            ax[idx].set_ylabel(r"$g-r$")
        ax[idx].legend(title=title, loc='upper center',
                       bbox_to_anchor=(0.5, -0.125), fontsize=12,
                       fancybox=True, shadow=True, ncol=2)

    plt.savefig(plot_file, bbox_inches="tight")
    plt.close()

################################################################################

if __name__ == "__main__":
    main()
