#!/usr/bin/env python

from __future__ import print_function
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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

def read_data(file, udg_only=True, field=None):
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

    # Converts NaNs to -1 for integer data-types
    int_features = ['NUM500', 'NUM1000', 'cz']
    for obj in dict_list:
        for feat in int_features:
            if obj.get(feat)=='NaN':
                obj[feat] = -1

    # Creates DataFrame from list of dictionaries.
    # Specifies the data type for each columns.
    df = pd.DataFrame.from_dict(dict_list).astype(
         {   'NAME':str,    'FIELD':str,      'TABLE':int,
               'ra':float,    'dec':float,       'cz':int,   'redshift':float,
           'sepMpc':float, 'sepDEG':float,   'NUM500':int,
               'Re':float,   'MUg0':float,      'b/a':float,    'n':float,
             'Mnuv':float,     'Mg':float,       'Mr':float,   'Mz':float,
            'NUV-g':float,  'NUV-r':float,    'NUV-z':float,   'UV':str,
              'g-r':float,    'g-z':float,      'r-z':float,
              'udg':str, 'LocalEnv':str,  'GlobalEnv':str,   'Density':str   } )

    # Specifies High Density Environment
    #df['DenseEnv'] = [obj['LocalEnv']=='Dense' or obj['GlobalEnv']=='Cluster'
    #                  for obj in df]

    # Filters out objects not in specified field
    if field:
        df = df.loc[df['FIELD']==field]

    # Return DataFrame with UDGs only if requested.
    if udg_only:
        return df.loc[df['udg']=='TRUE']

    return df

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

def get_label_color_marker(df, efeat="LocalEnv"):

    # Local Environment
    if efeat == "LocalEnv":
        label  = [ r'$\mathrm{Dense}$'         if  val=='Dense'  else \
                   r'$\mathrm{Sparse}$'        if  val=='Sparse' else \
                   r'$\mathrm{Unconstrained}$' for val in df[efeat] ]
        color  = [ 'lime'   if  val=='Dense'  else \
                   'orange' if  val=='Sparse' else \
                   'blue'   for val in df[efeat] ]
        marker = [ '^'      if  val=='Dense'  else \
                   'o'      if  val=='Sparse' else \
                   'x'      for val in df[efeat] ]
        legend_title = r"$\mathrm{Local \, Environment}$"
    
    # Global/Cluster Environment
    elif efeat == "GlobalEnv":
        label  = [ r'$\mathrm{Cluster}$'                if val=='Cluster'     else \
                   r'$\mathrm{Non}$-$\mathrm{Cluster}$' if val=='Non-Cluster' else \
                   r'$\mathrm{Unconstrained}$'          for val in df[efeat] ]
        color  = [ 'lime'   if  val=='Cluster'     else \
                   'orange' if  val=='Non-Cluster' else \
                   'blue'   for val in df[efeat] ]
        marker = [ '^'      if  val=='Cluster'     else \
                   'o'      if  val=='Non-Cluster' else \
                   'x'      for val in df[efeat] ]
        legend_title = r"$\mathrm{Coma \, Membership}$"
        
    # Environment Density
    elif efeat == "Density":
        label  = [ r'$\mathrm{High}$'           if val=='High' else \
                   r'$\mathrm{Low}$'            if val=='Low'  else \
                   r'$\mathrm{Unconstrained}$'  for val in df[efeat] ]
        color  = [ 'lime'   if  val=='High' else \
                   'orange' if  val=='Low'  else \
                   'blue'   for val in df[efeat] ]
        marker = [ '^'      if  val=='High' else \
                   'o'      if  val=='Low'  else \
                   'x'      for val in df[efeat] ]
        legend_title = r"$\mathrm{Density}$"
                   
    else:
        label  = [None] * len(df)
        color  = ['b']  * len(df)
        marker = ['x']  * len(df)
        legend_title = None
        
    return label, color, marker, legend_title
    
################################################################################

def color_plots(df, xfeat, yfeat, efeat="GlobalEnv", mfeat="Re",  flag="UV", plot_fname='color.pdf'):
    """
    Creates color-color or color-magnitude plots.
    
    df (DataFrame)
    xfeat (str): Color or Magnitude Feature
    yfeat (str): Color or Magnitude Feature
    efeat (str): Environment Feature (i.e., 'LocalEnv' or 'GlobalEnv')
    mfeat (str): Feature to base Marker Size
    flag  (str): Feature to Specify Detection ("Yes") or Upper Limit ("No")
    plot_fname (str): Filename of Plot
    """

    # Remove NaNs & Sorts Data to Plot Big Markers First
    df = df[[xfeat, yfeat, mfeat, efeat, flag]].dropna()
    df = df.sort_values(by=[mfeat], ascending=False)

    # Select Legend Labels, Marker Sizes & Colors & Shapes
    small_thres = 1.5  # kpc
    large_thres = 3.5  # kpc
    fontsize    = 30
    marker_size = 40
    marker_edge = 'k'
    thin_line   = 0.3
    thick_line  = 2.25
    label, color, marker, legend_title = get_label_color_marker(df, efeat)

    # Scatter Plot
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    
    for idx in range(len(df)):
        if df[flag].iloc[idx] == "No" and 'Mnuv' in yfeat:
            plt.arrow( df[xfeat].iloc[idx], df[yfeat].iloc[idx],
                       # Dictates Arrow Size/End point
                       dx=0, dy=-df[mfeat].iloc[idx]/max(df[mfeat]),
                       color = color[idx],
                       head_width = (df[mfeat].iloc[idx]/10)**2,
                       # Add Bold Outline around `mfeat` Values above `large_thres`
                       linewidth=thick_line if df[mfeat].iloc[idx]>large_thres else thin_line )
        else:
            plt.scatter( df[xfeat].iloc[idx],
                         df[yfeat].iloc[idx],
                         label  = label[idx],
                         color  = color[idx],
                         marker = marker[idx],
                         # Marker Radius Scales Linearly with `mfeat` Value
                         s      = marker_size * (df[mfeat].iloc[idx])**2,
                         edgecolors=marker_edge,
                         # Add Bold Outline around `mfeat` Values above `large_thres`
                         linewidth=thick_line if df[mfeat].iloc[idx]>large_thres else thin_line )

    plt.tick_params(which='both', direction='in', pad=10, labelsize=fontsize)
    plt.minorticks_on()
    
    xlabel = xfeat.replace('Mnuv','M_\mathrm{NUV}')
    plt.xlabel(('$' if '-' in xlabel else '$M_') + xlabel +'$', fontsize=fontsize)
    plt.ylabel('$'+ yfeat.replace('Mnuv','M_\mathrm{NUV}') +'$', fontsize=fontsize)
    plt.legend(title=legend_title)
    
    # Unique Markers in Legend Only (Uses Markers w/o Bold Outline)
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[::-1]
    labels  = labels[::-1]
    unique  = [ (h, l) for i, (h, l) in enumerate(zip(handles, labels)) \
                       if  l not in labels[:i] ]
    legend  = ax.legend( *zip(*unique), loc='lower right',
                         title_fontsize=24,
                         prop={'size':22},
                         fancybox=True,
                         frameon=True,
                         title=legend_title )
                        
    # Set Marker Size in Legend to `small_thres` Size
    for legend_handle in legend.legendHandles:
        legend_handle._sizes = [marker_size * small_thres**2]
    
    # Sets Axes Line Width
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
    
    # Removes Border Whitespace & Save
    plt.tight_layout()
    plt.savefig(plot_fname, format='pdf')
    plt.close()

################################################################################

def main(data_file='kadowaki2019.tsv',
         data_directory='../data',
         plot_directory='../plots',
         pair_name='pair.pdf',
         color_name=True,
         plot_pair=False,
         plot_statistical_tests=True,
         udg_only=False,
         local_env=True,
         density=False,
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

    # Data File
    param_file = os.path.join(data_directory, data_file)
    
    prefix = ('udgs'  if udg_only  else 'candidates') + '_' + \
             ('density' if density else ('local' if local_env else 'global')) + '_'

    # Save to Plots
    pair_plot = os.path.join(plot_directory, prefix + pair_name)
    
    three_colors = not udg_only and local_env

    global hist_color
    if three_colors:
        hist_color = 'Green'
    else:
        if hack_color_fix:
            change_color(three_colors=three_colors)

    ############################################################################

    df_results = read_data(param_file, udg_only=udg_only, field='Coma')
    efeat      = 'Density' if density else ('LocalEnv' if local_env else 'GlobalEnv')
    df_results = df_results.sort_values(by=[efeat])
    df_results = df_results.reset_index(drop=True)
    
    ############################################################################

    if color_name:
        
        color_features = ["NUV-r", "g-r", "g-z"]
        mag_features   = ["Mz"]
        
        for idx1,color in enumerate(color_features):
        
            # Color-Magnitude Plots
            for mag in mag_features:
                if mag not in color:
                    # File Name
                    cm_fname = os.path.join(plot_directory,
                               prefix + color + "_" + mag + ".pdf")
                    # Plot
                    color_plots(df_results,  xfeat=mag,    yfeat=color,
                                efeat=efeat, mfeat="Re", flag="UV",
                                plot_fname=cm_fname)
            
            # Color-Color Plots
            for idx2,color2 in enumerate(color_features):
                if (idx1 < idx2) and all([c not in color2 for c in color.split('-')]):
                    # File Name
                    cc_fname = os.path.join(plot_directory,
                               prefix + color + "_" + color2 + ".pdf")
                    # Plot
                    color_plots(df_results,  xfeat=color2, yfeat=color,
                                efeat=efeat, mfeat="Re", flag="UV",
                                plot_fname= cc_fname)
                        
                    
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
        features = ["cz", "MUg0", "Mg", "g-r", "Re", "b/a", "n", efeat]

        markers   = ['^',     'o']        if udg_only else ['^',      'o',       'x']
        col_list  = ['lime',   'Orange']  if udg_only else ['lime',  'Orange',  'Blue' ]
        cmap_list = ['Greens', 'Oranges'] if udg_only else ['Greens', 'Oranges', 'Blues']
        env_list  = sorted(df_results[efeat].unique())
        col_dict  = dict(zip(env_list, col_list))
        cmap_dict = dict(zip(env_list, cmap_list))

        ax = sns.PairGrid(data=df_results[features],
                          hue=efeat ,
                          palette=col_dict,
                          diag_sharey=False,
                          hue_kws={"marker":markers})

        ############################################################################
        
        def contours(*args,**kwargs):

            if verbose:
                print("\ncontours")
            new_args = remove_repeating_lists(remove_nan(args, verbose=verbose), verbose=verbose)
            
            if len(new_args) > 1:
                print(df_results[efeat])
                idx = args[0].index.values[0]
                label = df_results[efeat].iloc[idx]
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
        if density:
            env_replacements = {'High':r"$\mathrm{High}$",
                                'Low':r"$\mathrm{Low}$"}
        elif local_env:
            env_replacements = {'Dense':r"$\mathrm{Dense}$",
                                'Sparse':r"$\mathrm{Sparse}$"}
        else:
            env_replacements = {'Cluster':r"$\mathrm{Cluster}$",
                                'Non-Cluster':r"$\mathrm{Non}$-$\mathrm{Cluster}$"}
        if not udg_only:
            env_replacements["Unconstrained"] = r"$\mathrm{Unconstrained}$"

        # Replace Current Labels for LaTeX Labels
        labels = [env_replacements[env_label] for env_label in ax._legend_data.keys()]

        # LEGEND HANDLES
        handles = ax._legend_data.values()

        # ADD LEGEND & Fix Placement
        ax.fig.legend(handles=handles, labels=labels,
                      loc='lower center', ncol=3, fontsize=15,
                      frameon=True, edgecolor='k', markerscale=2.5,
                      title=r"$\mathrm{Environment \, Density}$" if density else \
                            r"$\mathrm{Local \, Environment}$" if local_env else \
                            r"$\mathrm{Coma \, Membership}$",
                      title_fontsize=20)
        ax.fig.subplots_adjust(top=1.05, bottom=0.12)


        # AXIS LABELS
        replacements = { # Magnitudes
                         "Mnuv":r'$M_\mathrm{NUV}$',
                         "Mg":r'$M_g$',
                         "Mr":r'$M_r$',
                         "Mz":r'$M_z$',

                         # Colors
                         "NUV-g":r'$\mathrm{NUV} - g$',
                         "g-r":r'$g - r$',
                         "r-z":r'$r - z$',

                         # Intrinsic Properties
                         "n":r'$n$',
                         "Re":r'$r_e \, \left( \mathrm{kpc} \right)$',
                         "MUg0":r'$\mu \left(g,0\right) \, \left( \mathrm{mag} \, \mathrm{arcsec}^{-2} \right)$',
                         "b/a":r'$b/a$',

                         # Extrinsic Properties
                         "cz":r'$cz \, \left( \mathrm{km/s} \right)$',
                         "sepMpc":r'$r_\mathrm{proj}$ \, \left( \mathrm{Mpc} \right)',
                         "NUM500":r'$\mathrm{\# \, of \, Massive \, Companions}$' }

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
        plt.close()


    ############################################################################

    if plot_statistical_tests:

        if density:
            df_sparse    = df_results.loc[df_results[efeat] == "Low"]
            df_dense     = df_results.loc[df_results[efeat] == "High"]
        elif local_env:
            df_sparse    = df_results.loc[df_results[efeat] == "Sparse"]
            df_dense     = df_results.loc[df_results[efeat] == "Dense"]
        else:
            df_sparse    = df_results.loc[df_results[efeat] == "Non-Cluster"]
            df_dense     = df_results.loc[df_results[efeat] == "Cluster"]

        feature_list = [ # Magnitudes
                           "Mnuv", "Mg", "Mr", "Mz",
                         # Colors
                           "NUV-g", "NUV-r", "NUV-z", "g-r", "g-z", "r-z",
                         # Intrinsic Properties
                           "n", "Re", "MUg0", "b/a",
                         # Extrinsic Properties
                           "cz", "sepMpc", "NUM500" ]


        for idx, feature in enumerate(feature_list):

            # Student's T-Test
            t_stat, t_pval = stats.ttest_ind(df_sparse[feature].dropna(),
                                             df_dense[feature].dropna(),
                                             equal_var=False)

            # Kolmoglov-Schmirnov Test
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
    main(plot_pair=True,
         color_name=True,
         plot_statistical_tests=False,
         udg_only=False,
         local_env=True,
         verbose=False,
         hack_color_fix=True)
    
    print("\n~~~~~~GLOBAL~~~~~~")
    main(plot_pair=True,
         color_name=True,
         plot_statistical_tests=False,
         udg_only=False,
         local_env=False,
         hack_color_fix=False)
    
    print("\n~~~~~~DENSITY~~~~~~")
    main(plot_pair=False,
         color_name=True,
         plot_statistical_tests=False,
         udg_only=False,
         density=True,
         hack_color_fix=False)
    
    
    print("\n-------------------- ALL UDGS --------------------")
    print("\n~~~~~~LOCAL~~~~~~")
    main(plot_pair=True,
         color_name=True,
         plot_statistical_tests=False,
         udg_only=True,
         local_env=True,
         hack_color_fix=False)
         
    print("\n~~~~~~GLOBAL~~~~~~")
    main(plot_pair=True,
         color_name=True,
         plot_statistical_tests=False,
         udg_only=True,
         local_env=False,
         hack_color_fix=False)

    print("\n~~~~~~DENSITY~~~~~~")
    main(plot_pair=False,
         color_name=True,
         plot_statistical_tests=False,
         udg_only=True,
         density=True,
         hack_color_fix=False)
