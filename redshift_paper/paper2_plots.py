#!/usr/bin/env python

from __future__ import print_function
import csv
import errno
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
import numpy as np
import os
import pandas as pd
import seaborn as sns
sns.set(font_scale=1.5, rc={'text.usetex' : True}, style="dark", color_codes=True)

################################################################################

class UDG:
    
    """
    UDG: Builds a Class for UDGs.
    """
    
    def __init__(self, name, ra, dec, angular_size, cz):
        self.name = name
        self.ra   = ra
        self.dec  = dec
        self.cz   = cz
        self.angular_size = angular_size

################################################################################

def get_physical_size(velocity, angular_size, H0=70):
    
    """
    GET_PHYSICAL_SIZE: Computes an object's physical size given its angular
                       size, it's recessional veloctiy, and an assumed
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
    """
    # SMUDGES Survey Data
    redshift_file = 'results.tsv'
    smudges_file  = 'smudges.txt'
    
    smudges = pd.read_table( os.path.join(directory, smudges_file),
                             skiprows=35,
                             delim_whitespace=True,
                             names=('SMDG_Name', 'ra', 'dec', 'mu_0_g', 'mu_0_r',
                                    'mu_0_z', 'err_mu_g', 'err_mu_r', 'err_mu_z',
                                    're', 'err_re', 'b/a', 'err_b/a', 'theta',
                                    'err_theta', 'M_g', 'M_r', 'M_z', 'err_Mg',
                                    'err_Mr', 'err_Mz' ))

    # Redshift Measurements
    results = pd.read_csv( os.path.join(directory, redshift_file), sep='\t')

    # cz-DataFrame
    df_cz = results[ ~pd.isna(results['cz']) ][['SMDG_Name','cz','cz_err']]
    """
################################################################################

def main(data_directory='.',
         plot_directory='.',
         data_file='kadowaki19_redshifts.tsv',
         update_dat_file=False,
         plot_pair=True,
         plot_hist=False):
    
    """
    Args:
        (str)  data_directory:
        (str)  plot_directory:
        (str)  data_file:
        (bool) update_dat_file:
    """


    ############################################################################

    coords_file    = os.path.join(data_directory, 'redshifts2.dat')
    results_file   = os.path.join(data_directory, data_file)
    redshift_plot  = os.path.join(plot_directory, 'cz_hist.pdf')
    axisratio_plot = os.path.join(plot_directory, 'reff_axisratio.pdf')
    pair_plot      = os.path.join(plot_directory, 'pair.pdf')

    # Load Data
    df_results = read_data(results_file)


    # Computes the Effictive Radius
    r_eff = get_physical_size(df_results['cz'], df_results['ComaSize'], H0=70)
    df_results['Reff'] = r_eff

    # Isolated
    iso = ["Isolated" if val==0 else "Too Close" if val==-1 else "Companions"
           for val in df_results['NUM500']]
    df_results['Environment'] = iso
    

    ############################################################################

    def remove_nan(args):

        nan_idx = np.array([])
        for a in args:
            # Ensures All List-like Data Structures are Pandas Series
            if type(a) == np.ndarray:
                a = pd.Series(a)
            print("\t", a.name)
            
            # Appends All Indices Corresponding to NaNs
            nan_idx = np.concatenate((nan_idx, a[pd.isna(a)].index.values))
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
    
    def remove_repeating_lists(args):
        
        new_args = []
        
        for a in args:
            if np.size(np.unique(a)) > 1:
                new_args.append(a)
    
        print(new_args)
        return new_args
    
    ############################################################################
    
    def get_color():
        pass
    
    ############################################################################
    
    def hist(*args,**kwargs):
        print("\nhist")
        
        new_args = remove_repeating_lists(remove_nan(args))
        large_args = []
        min_y, max_y = 9999999, -9999999
        
        for a in new_args:
            if len(a) > 10:
                large_args.append(a)
                
                if min(a) < min_y:
                    min_y = min(a)
                if max(a) > max_y:
                    max_y = max(a)
    
        if len(large_args):
            dist = sns.distplot(*large_args)
            axes = dist.axes
            axes.set_ylim(min_y, max_y)

    ############################################################################

    def scatter(*args,**kwargs):
        plt.scatter(*args, **kwargs, marker='.')
        
    ############################################################################

    if plot_pair:
    
        sns.set(style="ticks", color_codes=True)
        features = ["cz", "Reff", "MUg0", "Mg", "b/a", "NUM500", "Environment"]
        
        col_list  = ['Blue',  'Green',  'Red']
        cmap_list = ['Blues', 'Greens', 'Reds']
        env_list  = df_results["Environment"].unique()
        col_dict  = dict(zip(env_list, col_list))
        cmap_dict = dict(zip(env_list, cmap_list))
        
        colors = sns.color_palette("Set3", 3)
        
        ax = sns.PairGrid(data=df_results[features],
                          hue="Environment",
                          palette=col_dict)
        
        ############################################################################
        
        def contours(*args,**kwargs):
            print("\ncontours")
            new_args = remove_repeating_lists(remove_nan(args))
            
            if len(new_args) > 1:
                idx = args[0].index.values[0]
                label = df_results["Environment"].iloc[idx]
                cmap  = cmap_dict.get(label)
                print(idx, label, cmap)
                
                sns.kdeplot(*new_args, **kwargs, cmap=cmap, shade_lowest=True)
        
        ############################################################################
        
        
        ax.map_diag(hist)
        ax.map_lower(contours)
        ax.map_upper(scatter)
        ax.add_legend()
        

        replacements = {"cz":r'$cz \, \left( \mathrm{km/s} \right)$',
                        "Reff":r'$R_\mathrm{eff} \, \left( \mathrm{kpc} \right)$',
                        "MUg0":r'$\mu \left(g,0\right) \, \left( \mathrm{mag} \, \mathrm{arcsec}^{-2} \right)$',
                        "Mg":r'$M_g \, \left( \mathrm{mag} \right)$',
                        "b/a":r'$b/a$',
                        "NUM500":r'$\mathrm{\# \, of \, Massive \, Companions}$',
                        "Environment":r'$\mathrm{Environment}$'}

        for x_idx in range(len(features)-1):
            for y_idx in range(len(features)-1):
                
                xlabel = ax.axes[x_idx][y_idx].get_xlabel()
                ylabel = ax.axes[x_idx][y_idx].get_ylabel()
                
                if xlabel in replacements.keys():
                    ax.axes[x_idx][y_idx].set_xlabel(replacements[xlabel])
                if ylabel in replacements.keys():
                    ax.axes[x_idx][y_idx].set_ylabel(replacements[ylabel])
    
        # Save & Display Figure
        plt.savefig(pair_plot, bbox_inches = 'tight')


    ############################################################################
    
    if plot_hist:

        # HISTOGRAM OF RECESSIONAL VELOCITIES
        # Sturge's Rule on Number of Bins
        num_bins = 12 #int(np.ceil( 1 + 3.322 * np.log(np.size(df_results['cz'])) ))

        # Histogram
        ax = sns.distplot(df_results['cz'], bins=num_bins,
                          rug=True, kde=False,
                          axlabel=r"$cz \, \mathrm{(km/s)}$")
                          
        # Save & Display Figure
        fig = ax.get_figure()
        fig.savefig(redshift_plot, bbox_inches = 'tight')


        # Scatter Plot
        data_okay = ~df_results['b/a'].isna() * ~r_eff.isna()
        colors = sns.color_palette("hls", max(df_results['SOURCE']))
        cv = [colors[idx-1] + (1,) for idx in df_results['SOURCE'][data_okay].values]
        
        ax = sns.jointplot(r_eff, df_results['b/a'], space=0, joint_kws={"color":cv}).set_axis_labels( r"$r_\mathrm{eff} \, \mathrm{(kpc)}$", r"$b/a$")

        # Save & Display Figure
        ax.savefig(axisratio_plot, bbox_inches = 'tight')


    ############################################################################

    if update_dat_file:

        z = df_results['cz'].values / 299792
        coords_list = [convert_name_to_coord(n) + (round(z[idx],6),) for idx,n in enumerate(df_results['NAME'].values)]

        with open(coords_file, 'w') as f:
            for t in coords_list:
                f.write('\t'.join(str(s) for s in t) + '\n')

################################################################################

if __name__ == '__main__':
    main()
