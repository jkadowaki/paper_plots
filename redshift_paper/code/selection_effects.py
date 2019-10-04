#!/usr/bin/env python

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
import numpy as np
import os
import pandas
import seaborn as sns

from pair_plot import read_data


################################################################################

def plot_separation(xfeats = ["sepMpc", "Reff", "n", "MUg0", "g", "b/a", "g-z"],
                    data_fname='kadowaki2019.tsv',
                    plot_fname='SelectionEffect.pdf', udg_only=True,
                    data_directory='../data', plot_directory='../plots'):

    data_file = os.path.join(data_directory, data_fname)
    df = read_data(data_file, udg_only=udg_only, field='Coma')


    feature_list = [ # Magnitudes
                       "NUV", "g", "r", "z",
                     # Colors
                       "g-r", "g-z",
                     # Intrinsic Properties
                       "n", "Reff", "MUg0", "b/a",
                     # Extrinsic Properties
                       "cz", "NUM500" ]

    replacements = { # Magnitudes
                    "NUV":r'$\mathrm{NUV}$',
                    "g":r'$g$',
                    "r":r'$r$',
                    "z":r'$z$',

                    # Colors
                    "g-r":r'$g - r$',
                    "g-z":r'$g - z$',
            
                    # Intrinsic Properties
                    "n":r'$n$',
                    "Reff":r'$R_\mathrm{eff} \, \left( \mathrm{kpc} \right)$',
                    "MUg0":r'$\mu \left(g,0\right) \, \left( \mathrm{mag} \, \mathrm{arcsec}^{-2} \right)$',
                    "b/a":r'$b/a$',

                    # Extrinsic Properties
                    "cz":r'$cz \, \left( \mathrm{km/s} \right)$',
                    "sepDEG":r'$r_\mathrm{proj} \, \left( \mathrm{Deg} \right)$',
                    "sepMpc":r'$r_\mathrm{proj} \, \left( \mathrm{Mpc} \right)$',
                    "NUM500":r'$\mathrm{\# \, of \, Massive \, Companions}$' }

    box_length = 3
    num_row    = int(np.ceil(np.sqrt(len(feature_list))))
    num_col    = int(np.ceil(len(feature_list) / num_row))
    
    #"""
    
    prefix    = ('udgs'  if udg_only  else 'candidates') + '_'
    
    for xfeat in xfeats:
        plot_file = os.path.join(plot_directory, prefix + xfeat.replace("/","") + "_" + plot_fname)

    
        fig  = plt.subplots(  num_row,     num_col,
                              sharex=True, sharey=False,
                              figsize=(num_row*box_length,
                                       num_col*box_length)  )
        
        df2 = df[df["TABLE"]==2]
        df3 = df[df["TABLE"]==3]
        df4 = df[df["TABLE"]==4]
        
        for idx,yfeat in enumerate(feature_list):
            plt.subplot(num_row, num_col, idx+1, frameon=True)
            plt.scatter(df2[xfeat], df2[yfeat], marker='o', s=3, c='r', label="Table 2")
            plt.scatter(df3[xfeat], df3[yfeat], marker='^', s=3, c='g', label="Table 3")
            plt.scatter(df4[xfeat], df4[yfeat], marker='x', s=3, c='b', label="Table 4")
            plt.ylabel(replacements.get(yfeat))
        
        plt.subplots_adjust(hspace=0.2, wspace=0.4)
        plt.xlabel(replacements.get(xfeat))
        plt.legend()
        
        plt.savefig(plot_file, bbox_inches = 'tight')
        plt.close()
    
    """
    
    g = sns.pairplot(df, x_vars=[xfeat], y_vars=feature_list,
                     hue="TABLE", height=5, aspect=1, kind="reg")
                     
    for idx in range(len(feature_list)):
    
        # Change Axis Limits
        g.axes[idx][0].set_xlim(df[xfeat].min(), df[xfeat].max())
        g.axes[idx][0].set_ylim(df[feature_list[idx]].min(),
                                                df[feature_list[idx]].max())
        
        # Format Axis Labels
        g.axes[idx][0].tick_params(labelsize=15)

        xlabel = g.axes[idx][0].get_xlabel()
        ylabel = g.axes[idx][0].get_ylabel()
        
        if xlabel in replacements.keys():
            g.axes[idx][0].set_xlabel(replacements[xlabel], fontsize=20)
        if ylabel in replacements.keys():
            g.axes[idx][0].set_ylabel(replacements[ylabel], fontsize=20)

    # Legend Placement
    handles = g._legend_data.values()
    labels = g._legend_data.keys()
    g.fig.legend(handles=handles, labels=labels, loc='lower center', ncol=3,
                 title=r"$\mathrm{Table}$", title_fontsize=20)
    g.fig.subplots_adjust(top=1.05, bottom=0.02)
    
    plot_file = os.path.join(plot_directory, prefix + plot_fname)
    
    plt.savefig(plot_file, bbox_inches = 'tight')
    plt.close()

    """
    
    
################################################################################

if __name__ == "__main__":
    
    plot_separation(udg_only=True)
    plot_separation(udg_only=False)
