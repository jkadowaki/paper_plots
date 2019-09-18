#!/usr/bin/env python

from __future__ import print_function
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import csv
import errno
import matplotlib
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
import numpy as np
import os
import pandas as pd
from scipy import stats
import seaborn as sns

# GLOBAL VARIABLES
hist_color = 'Green'
hist_idx   = -1

################################################################################
"""
PAIR_PLOT.py
    Goal: Creates
    
    Methods:
    (1)
    (2)
    (3)
"""
################################################################################

def convert(str):
    """
    
    ARGS:
    RETURNS:
    """
    try:
        return int(str)
    except:
        return float(str)
    finally:
        return str

################################################################################

def read_data(file, udgs_only=True):
    """
    Loads relevant data from file into a Pandas DataFrame.
    
    ARGS:
        file (str): Name of .tsv file containing UDG properties.
    RETURNS:
        (DataFrame):
    """

    dict_list = []
    columns = []

    with open(file) as f:
        for idx, line in enumerate(f):
            if idx == 0:
                columns = line.split()
            else:
                dict_list.append( {key:val for key,val
                                    in zip(columns, line.split())} )

    # Creates DataFrame from list of dictionaries.
    # Specifies the data type for each columns.
    df = pd.DataFrame.from_dict(dict_list).astype(
            {'NAME':str,     'udg':str,   'ra':float,  'dec':float,  'redshift':float,
             'Reff':float,   'b/a':float, 'cz':int,    'n':float,    'NUM500':int,
             'NUV':float,    'g':float,   'r':float,   'z':float,
             'NUV-g':float,  'g-r':float, 'r-z':float, 'MUg0':float,
             'LocalEnv':str, 'GlobalEnv':str} )

    # Return DataFrame with UDGs only if requested.
    if udg_only:
        return df[df.loc[df['udg']=='TRUE']]

    return df


################################################################################

def convert_name_to_coord(name):
    #
    n=2 # split string after n characters
    positive = True
    #
    try:
        ra, dec = name.strip('SMDG').split('+')
    except:
        ra, dec  = name.strip('SMDG').split('-')
        positive = False
    #
    ra_hour, ra_min, ra_sec, ra_p = [int(ra[i:i+n]) for i in range(0, len(ra), n)]
    dec_dec, dec_min, dec_sec = [int(dec[i:i+n]) for i in range(0, len(dec), n)]
    #
    new_ra = (int(ra_hour)/24. + int(ra_min)/24./60. + float(ra_sec + ra_p)/24./3600.) * 360.
    new_dec = int(dec_dec) + int(dec_min)/60. + float(dec_sec)/3600.
    #
    if positive:
        return np.round(new_ra, 6), np.round(new_dec, 6)
    #
    return np.round(new_ra, 6), -np.round(new_dec, 6)

################################################################################

def change_color(three_colors=False):
    
    global hist_color
    
    if not three_colors:
        if hist_color == 'Orange':
            hist_color = 'Green'
        else:
            hist_color = 'Orange'
    else:
        if hist_color == 'Green':
            hist_color = 'Orange'
        elif hist_color == 'Orange':
            hist_color = 'Blue'
        else:
            hist_color = 'Green'

################################################################################

def color_plots(df, xfeat, yfeat, environment_scale, local_env,
                ms_feat="Reff", ms=10,
                xlabel="$\mathrm{NUV} - g$",
                ylabel="$g-r$",
                plot_fname="color_color.pdf"):
    
    if local_env:
        df_sparse    = df.loc[df[environment_scale] == "Sparse"]
        df_dense     = df.loc[df[environment_scale] == "Dense"]
    else:
        df_sparse    = df.loc[df[environment_scale] == "Non-Member"]
        df_dense     = df.loc[df[environment_scale] == "Member"]

    #Remove NaNs
    df_sparse = df_sparse[[xfeat, yfeat, ms_feat]].dropna()
    df_dense  = df_dense[[xfeat, yfeat, ms_feat]].dropna()

    # PLOT
    fig = plt.figure()

    print("\n{0}\nNUV-g:\n\t".format("SPARSE" if local_env else "Non-Member"),
          df_sparse[[xfeat, yfeat, ms_feat]])
    
    plt.scatter(df_sparse[[xfeat]], df_sparse[[yfeat]],
                s=ms*df_sparse[[ms_feat]], color='orange', marker='o',
                label=r"$\mathrm{Sparse}$" if local_env else r"$\mathrm{Non}$-$\mathrm{Member}$")

    print("\n{0}\nNUV-g:\n\t".format("DENSE" if local_env else "Member"),
          df_dense[[xfeat, yfeat, ms_feat]])

    plt.scatter(df_dense[[xfeat]], df_dense[[yfeat]],
                s=ms*df_dense[[ms_feat]], color='lime', marker='^',
                label=r"$\mathrm{Dense}$" if local_env else r"$\mathrm{Member}$")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title=r"$\mathrm{Local \, Environment}$" if local_env else r"$\mathrm{Coma \, Membership}$")
    plt.tight_layout()
    plt.savefig(plot_fname, format='pdf')



################################################################################

def main(data_directory='.',
         plot_directory='.',
         pair_name='pair.pdf',
         cumsum_name='cumsum_all.pdf',
         color_name='color_color.pdf',
         update_dat_file=False,
         plot_pair=False,
         plot_statistical_tests=True,
         udg_only=False,
         local_env=True,
         verbose=False,
         hack_color_fix=False):
    
    """
    Args:
        (str)  data_directory:
        (str)  plot_directory:
        (str)
        (bool) update_dat_file:
    """

    ############################################################################

    # Load Appropriate Parameter File
    udg_param_file = os.path.join(data_directory,
                     'UDG2019_all_parameters.tsv' if udg_only else
                     'candidates2019_all_parameters.tsv')

    # Save to Appropriate Coordinate File
    if udg_only:
        if local_env: coords='redshifts2_local_udgs.dat'
        else:         coords='redshifts2_global_udgs.dat'
    else:
        if local_env: coords='redshifts2_local_candidates.dat'
        else:         coords='redshifts2_global_candidates.dat'
    coords_file = os.path.join(data_directory, coords)

    # Save to Plots
    pair_plot      = os.path.join(plot_directory, pair_name)
    ks_plot        = os.path.join(plot_directory, cumsum_name)

    three_colors = not udg_only and local_env

    global hist_color
    if three_colors:
        hist_color = 'Green'
    else:
        if hack_color_fix:
            change_color(three_colors=three_colors)

    ############################################################################

    df_results        = read_data(udg_param_file)
    environment_scale = 'Environment' if local_env else 'Member'
    df_results        = df_results.sort_values(by=[environment_scale])
    df_results        = df_results.reset_index(drop=True)

    ############################################################################

    if color_plots:
    
        f = os.path.splitext(color_name)
        color_plots(df_results,
                    xfeat="NUVg",
                    yfeat="gr",
                    ms_feat="Reff", ms=20,
                    local_env=local_env,
                    environment_scale=environment_scale,
                    plot_fname="".join(f[0]+"_Reff"+f[1]))
                
        color_plots(df_results,
                    xfeat="NUVg",
                    yfeat="gr",
                    ms_feat="b/a", ms=50,
                    local_env=local_env,
                    environment_scale=environment_scale,
                    plot_fname="".join(f[0]+"_ba"+f[1]))
                    
                    
    ############################################################################

    def remove_nan(args, verbose=False):

        nan_idx = np.array([])
        for a in args:
            # Ensures All List-like Data Structures are Pandas Series
            if type(a) == np.ndarray:
                a = pd.Series(a)
            if verbose:
                print("\t", a.name)
            
            # Appends All Indices Corresponding to NaNs
            nan_idx = np.concatenate((nan_idx, a[pd.isna(a)].index.values))
        if verbose:
            print(nan_idx)
        
        # Stores Arguments with NaNs Removed
        new_args = []
        
        for a in args:
            # Ensures All List-like Data Structures are Pandas Series
            if type(a) == np.ndarray:
                a = pd.Series(a)
            
            new_args.append( a.drop(nan_idx, errors="ignore") )
        
        return new_args
    
    ############################################################################
    
    def remove_repeating_lists(args, verbose=False):
        
        new_args = []
        
        for a in args:
            if np.size(np.unique(a)) > 1:
                new_args.append(a)

        if verbose:
            print(new_args)

        return new_args


    ############################################################################

    def hist(*args, **kwargs):
        if verbose:
            print("\nhist")

        new_args = remove_repeating_lists(remove_nan(args, verbose=verbose), verbose=verbose)
        large_args = []
        min_y, max_y = 9999999, -9999999
        
        for a in new_args:
            if len(a) > 4:
                large_args.append(a)
                
                if min(a) < min_y:
                    min_y = min(a)
                if max(a) > max_y:
                    max_y = max(a)

        if verbose:
            print(large_args)

        if len(large_args):

            hist_bins = np.linspace(min(a), max(a), 6)
            dist = sns.distplot(*large_args, rug=True, kde=True, hist=False, norm_hist=True, color=hist_color,  bins=hist_bins)
            sns.distplot(*large_args, kde=False, hist=True, norm_hist=True, color=hist_color, bins=hist_bins)

            axes      = dist.axes
            hist_val  = np.histogram(*large_args, bins=hist_bins, density=True)[0]
            ylimit    = np.max(hist_val)
            curr_ylim = axes.get_ylim()[1]

            if curr_ylim > 5*ylimit or ylimit > curr_ylim:
                axes.set_ylim(0, ylimit/0.8)

            axes.xaxis.set_tick_params(labelsize=50)
            axes.yaxis.set_tick_params(labelsize=50)

            change_color(three_colors = not udg_only and local_env)

    ############################################################################

    def scatter(*args,**kwargs):
        plt.scatter(*args, **kwargs, s=24, edgecolor='k', linewidth=0.1)
        
    ############################################################################

    if plot_pair:
    
        sns.set(style="ticks", color_codes=True)
        features = ["NUV",
                    #"g",
                    #"r",
                    #"NUVg",
                    #"gr",
                    "cz",
                    "Reff", "MUg0", "b/a",
                    "NUM500", environment_scale]

        markers   = ['^',     'o']        if udg_only else ['^',      'o',       'x']
        col_list  = ['lime',   'Orange']  if udg_only else ['lime',  'Orange',  'Blue' ]
        cmap_list = ['Greens', 'Oranges'] if udg_only else ['Greens', 'Oranges', 'Blues']
        env_list  = sorted(df_results[environment_scale].unique())
        col_dict  = dict(zip(env_list, col_list))
        cmap_dict = dict(zip(env_list, cmap_list))

        ax = sns.PairGrid(data=df_results[features],
                          hue=environment_scale ,
                          palette=col_dict,
                          diag_sharey=False,
                          hue_kws={"marker":markers})

        ############################################################################
        
        def contours(*args,**kwargs):

            if verbose:
                print("\ncontours")
            new_args = remove_repeating_lists(remove_nan(args, verbose=verbose), verbose=verbose)
            
            if len(new_args) > 1:
                idx = args[0].index.values[0]
                label = df_results[environment_scale].iloc[idx]
                cmap  = cmap_dict.get(label)
                if verbose:
                    print(idx, label, cmap)

                if idx != 1:  # Exclude Unconstrained
                    sns.kdeplot(*new_args, cmap=cmap, shade_lowest=True)
        
        ############################################################################

        ax.map_diag(hist)
        ax.map_lower(contours)
        ax.map_upper(scatter)


        # LEGEND LABELS
        if local_env:
            env_replacements = {'Dense':r"$\mathrm{Dense}$",   'Sparse':r"$\mathrm{Sparse}$"}
        else:
            env_replacements = {'Member':r"$\mathrm{Cluster}$", 'Non-Member':r"$\mathrm{Non}$-$\mathrm{Cluster}$"}
        if not udg_only:
            env_replacements["Unconstrained"] = r"$\mathrm{Unconstrained}$"

        # Replace Current Labels for LaTeX Labels
        labels = [env_replacements[env_label] for env_label in ax._legend_data.keys()]

        # LEGEND HANDLES
        handles = ax._legend_data.values()

        # ADD LEGEND & Fix Placement
        ax.fig.legend(handles=handles, labels=labels, loc='lower center', ncol=3, fontsize=15,
                      frameon=True, edgecolor='k', markerscale=2.5,
                      title=r"$\mathrm{Local \, Environment}$" if local_env else r"$\mathrm{Coma \, Membership}$",
                      title_fontsize=20)
        ax.fig.subplots_adjust(top=1.05, bottom=0.12)


        # AXIS LABELS
        replacements = {"NUV":r'$M_\mathrm{NUV}$',
                        "g":r'$g$',
                        "r":r'$r$',
                        "NUVg":r'$\mathrm{NUV} - g$',
                        "gr":r'$g - r$',
                        "cz":r'$cz \, \left( \mathrm{km/s} \right)$',
                        "Reff":r'$R_\mathrm{eff} \, \left( \mathrm{kpc} \right)$',
                        "MUg0":r'$\mu \left(g,0\right) \, \left( \mathrm{mag} \, \mathrm{arcsec}^{-2} \right)$',
                        "b/a":r'$b/a$',
                        "NUM500":r'$\mathrm{\# \, of \, Massive \, Companions}$'}
        #"Environment":r'$\mathrm{Environment}$'}

        for x_idx in range(len(features)-1):
            for y_idx in range(len(features)-1):

                ax.axes[x_idx][y_idx].tick_params(labelsize=15)

                xlabel = ax.axes[x_idx][y_idx].get_xlabel()
                ylabel = ax.axes[x_idx][y_idx].get_ylabel()
                
                if xlabel in replacements.keys():
                    ax.axes[x_idx][y_idx].set_xlabel(replacements[xlabel], fontsize=20)
                if ylabel in replacements.keys():
                    ax.axes[x_idx][y_idx].set_ylabel(replacements[ylabel], fontsize=20)


        # Save & Display Figure
        plt.savefig(pair_plot, bbox_inches = 'tight')


    ############################################################################

    if update_dat_file:
        
        with open(coords_file, 'w') as f:
            f.write(df_results[['ra', 'dec', 'z', environment_scale, 'Reff']].to_string(index=False, header=False))


    ############################################################################

    if plot_statistical_tests:

        if local_env:
            df_sparse    = df_results.loc[df_results[environment_scale] == "Sparse"]
            df_dense     = df_results.loc[df_results[environment_scale] == "Dense"]
        else:
            df_sparse    = df_results.loc[df_results[environment_scale] == "Non-Member"]
            df_dense     = df_results.loc[df_results[environment_scale] == "Member"]
        feature_list = ["NUV",
                        "g",
                        "r",
                        "NUVg",
                        "gr",
                        "cz",
                        "Reff", "MUg0", "b/a",
                        "NUM500"]

        box_len = 3
        num_row = 2
        num_col = int(np.ceil( len(feature_list)/num_row))


        #fig, axes = plt.subplots(num_row, num_col, figsize=(num_row * box_len, num_col * box_len)) #, subplot_kw=dict(polar=True))
        #   #g = sns.FacetGrid(df_results, col=feature_list, height=4)

        for idx, feature in enumerate(feature_list):

            t_stat, t_pval = stats.ttest_ind(df_sparse[feature].dropna(),
                                             df_dense[feature].dropna(),
                                             equal_var=False)

            ks_stat, ks_pval = stats.ks_2samp(df_sparse[feature].dropna(),
                                              df_dense[feature].dropna(),
                                              alternative="two-sided")

            print("\nFeature:",       feature)
            print("T-statistic:",     t_stat)
            print("P-value (2tail):", t_pval)
            print("KS-statistic:",    ks_stat)
            print("P-value (2tail):", ks_pval)

################################################################################

if __name__ == '__main__':

    
    print("\n----------------- ALL CANDIDATES -----------------")
    print("\n~~~~~LOCAL~~~~~~")
    main(plot_pair=False, plot_statistical_tests=True,
         pair_name='pair_all_local.pdf', cumsum_name='cumsum_all_local.pdf',
         color_name='color_all_local.pdf',
         udg_only=False, local_env=True, update_dat_file=True, verbose=False)
    
    print("\n~~~~~~GLOBAL~~~~~~")
    main(plot_pair=False, plot_statistical_tests=True,
         pair_name='pair_all_global.pdf', cumsum_name='cumsum_all_global.pdf',
         color_name='color_all_global.pdf',
         udg_only=False, local_env=False, update_dat_file=True,
         hack_color_fix=True)
    
    print("\n-------------------- ALL UDGS --------------------")
    print("\n~~~~~~LOCAL~~~~~~")
    main(plot_pair=False, plot_statistical_tests=True,
         pair_name='pair_udgs_local.pdf', cumsum_name='cumsum_udgs_local.pdf',
         color_name='color_udgs_local.pdf',
         udg_only=True, local_env=True, update_dat_file=True,
         hack_color_fix=False)
         
    print("\n~~~~~~GLOBAL~~~~~~")
    main(plot_pair=False, plot_statistical_tests=True,
         pair_name='pair_udgs_global.pdf', cumsum_name='cumsum_udgs_global.pdf',
         color_name='color_udgs_global.pdf',
         udg_only=True, local_env=False, update_dat_file=True,
         hack_color_fix=True)
