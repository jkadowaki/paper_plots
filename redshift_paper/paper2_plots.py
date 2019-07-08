#!/usr/bin/env python

from __future__ import print_function
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import csv
import errno
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
import numpy as np
import os
import pandas as pd
from scipy import stats
import seaborn as sns
sns.set(font_scale=25, rc={'text.usetex':True}, style="darkgrid", color_codes=True)
sns.set_style("darkgrid", {"legend.frameon":True})

hist_color = 'Blue'
hist_idx   = -1


################################################################################

def get_physical_size(velocity, angular_size, H0=70):
    """
    GET_PHYSICAL_SIZE: Computes an object's physical size given its angular
                       size, its recessional veloctiy, and an assumed
                       Hubble's constant.

    ARGS:
        (float) angular_size: Object's Angular Size in Arcseconds (")
        (float) velocity: Recessional Velocity in (km/s) Attributed to the
                          Universe's Expansion Rate
        (float) H0: Hubble's Constant in (km/s)/Mpc

    RETURNS:
        (float) physical_size: Object's Physical Size in kpc
    """

    RAD_TO_ARCSEC = 206265  # arcseconds in a radian
    return 1000 * velocity * angular_size / H0 / RAD_TO_ARCSEC


################################################################################

def get_absolute_magnitude(magnitude, velocity, H0=70):
    """
    GET_ABSOLUTE_MAGNITUDE: Computes an object's absolute magnitude given
                       its apparent magnitude, its recessional veloctiy, and an
                       assumed Hubble's constant.


    ARGS:
        (float) magnitude: Object's Apparent Magnitude
        (float) velocity: Recessional Velocity in (km/s) Attributed to the
                          Universe's Expansion Rate
        (float) H0: Hubble's Constant in (km/s)/Mpc

    RETURNS:
        (float) absolute_magnitude: Object's Physical Size in kpc
    """

    return magnitude - 5 * np.log10(velocity / H0) - 25


################################################################################

def convert(str):

    try:
        return int(str)
    except:
        return float(str)
    finally:
        return str

################################################################################

def read_data(file):

    dict_list = []
    columns = []
    #types = []

    with open(file) as f:
        for idx, line in enumerate(f):
            if idx == 0:
                columns = line.split()
            else:
                dict_list.append( {key:val for key,val
                                    in zip(columns, line.split())} )

    return pd.DataFrame.from_dict(dict_list).astype(
        {'ALTNAME':str, 'FIELD':str, 'MUg0':float, 'Mg':float, 'NAME':str,
         'ComaSize':float, 'SOURCE':int, 'b/a':float, 'cz':int, 'czerr':float,
         'NUM500':int, 'NUM1000':int} )

################################################################################

def convert_name_to_coord(name):
    
    n=2 # split string after n characters
    positive = True
    
    try:
        ra, dec = name.strip('SMDG').split('+')
    except:
        ra, dec  = name.strip('SMDG').split('-')
        positive = False

    ra_hour, ra_min, ra_sec, ra_p = [int(ra[i:i+n]) for i in range(0, len(ra), n)]
    dec_dec, dec_min, dec_sec = [int(dec[i:i+n]) for i in range(0, len(dec), n)]

    new_ra = (int(ra_hour)/24. + int(ra_min)/24./60. + float(ra_sec + ra_p)/24./3600.) * 360.
    new_dec = int(dec_dec) + int(dec_min)/60. + float(dec_sec)/3600.

    if positive:
        return round(new_ra, 6), round(new_dec, 6)

    return round(new_ra, 6), -round(new_dec, 6)


################################################################################

def main(data_directory='.',
         plot_directory='.',
         data_file='kadowaki19_redshifts.tsv',
         pair_name='pair.pdf',
         update_dat_file=False,
         plot_pair=False,
         plot_statistical_tests=True,
         udg_only=False,
         verbose=False):
    
    """
    Args:
        (str)  data_directory:
        (str)  plot_directory:
        (str)  data_file:
        (bool) update_dat_file:
    """


    ############################################################################


    processed_file = 'UDG2019_all_parameters.tsv' if udg_only else 'candidates2019_all_parameters.tsv'
    coords_file    = os.path.join(data_directory, 'redshifts2.dat')
    results_file   = os.path.join(data_directory, data_file)
    udg_param_file = os.path.join(data_directory, processed_file)
    redshift_plot  = os.path.join(plot_directory, 'cz_hist.pdf')
    axisratio_plot = os.path.join(plot_directory, 'reff_axisratio.pdf')
    pair_plot      = os.path.join(plot_directory, pair_name)
    ks_plot        = os.path.join(plot_directory, 'ks_test.pdf')

    # Load Data
    df_results = read_data(results_file)


    # Computes the Effictive Radius
    r_eff = get_physical_size(df_results['cz'], df_results['ComaSize'], H0=70)
    df_results['Reff'] = r_eff.round(decimals=2)
    if udg_only:
        df_results = df_results[df_results['Reff'] > 1.5]

    # Environment
    iso = ["Isolated" if val==0 else "Unconstrained" if val==-1 else "Interacting"
           for val in df_results['NUM500']]
    df_results['Environment'] = iso

    # Absolute Magnitude
    absmag = get_absolute_magnitude(df_results['Mg'], df_results['cz'], H0=70)
    df_results['absmag'] = absmag.round(decimals=2)

    df_results.to_csv(udg_param_file, index=False, sep='\t')
    df_results = df_results.sort_values(by=['Environment'])
    df_results = df_results.reset_index(drop=True)

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

    def change_color():

        global hist_color

        if hist_color == 'Red':
            hist_color = 'Blue'
        else:
            hist_color = 'Red'


    def hist(*args, **kwargs):
        if verbose:
            print("\nhist")

        new_args = remove_repeating_lists(remove_nan(args, verbose=verbose), verbose=verbose)
        large_args = []
        min_y, max_y = 9999999, -9999999
        
        for a in new_args:
            if len(a) > 10:
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

            change_color()

    ############################################################################

    def scatter(*args,**kwargs):
        plt.scatter(*args, **kwargs, s=15, edgecolor='k', linewidth=0.001)
        
    ############################################################################

    if plot_pair:
    
        sns.set(style="ticks", color_codes=True)
        features = ["cz", "Reff", "MUg0", "absmag", "b/a", "NUM500", "Environment"]

        markers   = ['^',     'o']    if udg_only else ['^', 'x', 'o']
        col_list  = ['Blue',  'Red']  if udg_only else ['Blue',  'Green',  'Red']
        cmap_list = ['Blues', 'Reds'] if udg_only else ['Blues', 'Greens', 'Reds']
        env_list  = sorted(df_results["Environment"].unique())
        col_dict  = dict(zip(env_list, col_list))
        cmap_dict = dict(zip(env_list, cmap_list))

        ax = sns.PairGrid(data=df_results[features],
                          hue="Environment",
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
                label = df_results["Environment"].iloc[idx]
                cmap  = cmap_dict.get(label)
                if verbose:
                    print(idx, label, cmap)

                if idx != 1:  # Exclude Unconstrained
                    sns.kdeplot(*new_args, **kwargs, cmap=cmap, shade_lowest=True)
        
        ############################################################################


        ax.map_diag(hist)
        ax.map_lower(contours)
        ax.map_upper(scatter)


        # LEGEND LABELS
        env_replacements = {"Interacting":r"$\mathrm{Interacting}$",
                            "Isolated":r"$\mathrm{Isolated}$",
                            "Unconstrained":r"$\mathrm{Unconstrained}$"}

        # Replace Current Labels for LaTeX Labels
        labels = [env_replacements[env_label] for env_label in ax._legend_data.keys()]

        # LEGEND HANDLES
        handles = ax._legend_data.values()

        # ADD LEGEND & Fix Placement
        ax.fig.legend(handles=handles, labels=labels, loc='lower center', ncol=3, fontsize=15,
                      frameon=True, edgecolor='k', markerscale=2.5,
                      title=r'$\mathrm{Environment}$', title_fontsize=20)
        ax.fig.subplots_adjust(top=1.05, bottom=0.12)


        # AXIS LABELS
        replacements = {"cz":r'$cz \, \left( \mathrm{km/s} \right)$',
                        "Reff":r'$R_\mathrm{eff} \, \left( \mathrm{kpc} \right)$',
                        "MUg0":r'$\mu \left(g,0\right) \, \left( \mathrm{mag} \, \mathrm{arcsec}^{-2} \right)$',
                        "absmag":r'$M_g \, \left( \mathrm{mag} \right)$',
                        "b/a":r'$b/a$',
                        "NUM500":r'$\mathrm{\# \, of \, Massive \, Companions}$',
                        "Environment":r'$\mathrm{Environment}$'}

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

        z = df_results['cz'].values / 299792
        coords_list = [convert_name_to_coord(n) + (round(z[idx],6),) for idx,n in enumerate(df_results['NAME'].values)]

        with open(coords_file, 'w') as f:
            for t in coords_list:
                f.write('\t'.join(str(s) for s in t) + '\n')


    ############################################################################

    if plot_statistical_tests:

        df_isolated    = df_results.loc[df_results["Environment"] == "Isolated"]
        df_interacting = df_results.loc[df_results["Environment"] == "Interacting"]

        #fig, axes = plt.subplots(2, 3) #, subplot_kw=dict(polar=True))

        for idx, feature in enumerate(["cz", "Reff", "MUg0", "Mg", "b/a", "NUM500"]):

            t_stat, t_pval = stats.ttest_ind(df_isolated[feature].dropna(),
                                             df_interacting[feature].dropna(),
                                             equal_var=False)

            ks_stat, ks_pval = stats.ks_2samp(df_isolated[feature].dropna(),
                                              df_interacting[feature].dropna(),
                                              alternative="two-sided")

            print("\nFeature:",       feature)
            print("T-statistic:",     t_stat)
            print("P-value (2tail):", t_pval)
            print("KS-statistic:",    ks_stat)
            print("P-value (2tail):", ks_pval)

            #axes[int(np.floor(idx/2)), idx%3].plot(stats.cumfreq) =
            #axes[int(np.floor(idx/2)), idx%3].set_title(feature)


################################################################################

if __name__ == '__main__':

    print("\n----------------- ALL CANDIDATES -----------------")
    main(plot_pair=True, plot_statistical_tests=True, pair_name='pair_all.pdf')

    print("\n-------------------- ALL UDGS --------------------")
    main(plot_pair=True, plot_statistical_tests=True, pair_name='pair_udgs.pdf', udg_only=True)

    # Need to implement plot!
