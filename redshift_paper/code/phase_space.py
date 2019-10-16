#! /usr/bin/env python

################################################################################

from conversions import get_angular_size, get_physical_size
import numpy as np
import matplotlib.pyplot as plt
from pair_plot import get_label_color_marker, read_data
import os


# CONSTANTS
global coma_ra, coma_dec, coma_z, c, h0, r_splash
coma_ra  = 194.953
coma_dec = 27.981
coma_z   = 0.0231
c        = 299792    # km/s
h0       = 67.37     # km/s/Mpc (Planck)
r_splash = 2.428     # Mpc


###############################################################################

def load_NED_data(fname):
    """
    """
    
    # NED SEARCH RESULTS:
    # 0  No.               6  Redshift                12  Photometry Points
    # 1  Object Name       7  Redshift Flag           13  Positions
    # 2  RA(deg)           8  Magnitude and Filter    14  Redshift Points
    # 3  DEC(deg)          9  Distance (arcmin)       15  Diameter Points
    # 4  Type             10  References              16  Associations
    # 5  Velocity         11  Notes
    
    results = np.genfromtxt(fname, skip_header=24, delimiter='\t',
                            dtype=None, encoding='ascii')
    
    velocity = [coma_galaxy[5] for coma_galaxy in results]
    distance = [coma_galaxy[9] for coma_galaxy in results]

    return velocity, distance


###############################################################################

def phase_space_plot(data_fname="kadowaki2019.tsv",
                     ned_fname="objsearch_cz2000-12000_500arcmin.txt",
                     plot_fname="phasespace.pdf",
                     local_env=True,
                     udg_only=True,
                     mfeat="Reff"):

    """
    Creates phase space diagram
    
    ARGS:
    
    RETURNS:
    
    """
    
    coma_vel, coma_dist = load_NED_data(ned_fname)
    
    efeat = 'LocalEnv' if local_env else 'GlobalEnv'
    
    df = read_data(data_fname, udg_only=udg_only, field="Coma")
    df = df[["ra", "dec", "cz", mfeat, efeat]].dropna()
    df = df.sort_values(by=[mfeat], ascending=False)
    
    # Sort UDGs to Plot Largest First & Smallest Last
    separation = get_angular_size(df["ra"], df["dec"], coma_ra, coma_dec)/60 # arcmin

    # Select Legend Labels, Marker Sizes & Colors & Shapes
    marker_size = 40  # Size of Marker for R_e = 1.0 kpc
    small_thres = 1.5 # kpc
    large_thres = 3.5 # kpc
    label_size  = 30
    label, color, marker, legend_title = get_label_color_marker(df, efeat)
    
    # Plot Limits
    kms_min     = 2000  if udg_only else 0
    kms_max     = 12050 if udg_only else 13000
    arcmin_min  = 0
    arcmin_max  = 505   if udg_only else 650
    mpc_min     = 0
    mpc_max     = get_physical_size(60*arcmin_max, c*coma_z, H0=h0)
    
    # Plot Phase Space Data
    legend_loc = 'lower right' if udg_only else 'upper right'

    
    
    # Create Figure
    plt.clf()
    plt.rcParams['savefig.facecolor'] = "1."
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size='28')

    # Establish Axis Limits & Labels
    fig, ax1 = plt.subplots(figsize=(10,10))
    ax1.set_xlim(mpc_min, mpc_max)
    ax1.set_xlabel("$r_\mathrm{proj} \, (\mathrm{Mpc})$",
                   size=label_size)
    ax1.set_ylabel("$cz \, (\mathrm{km \, s^{-1}})$", size=label_size)
    
    # Plot Splashback Radius
    ax1.plot((r_splash,r_splash), [kms_min, kms_max], 'r--', linewidth=3)
    
    plt.minorticks_on()
    plt.tick_params(which='both', direction='in', pad=10, width=1.5)
    ax1.tick_params(which='major', length=5)
    ax1.tick_params(which='minor', length=2)
    ax1.xaxis.set_ticks(np.arange(0,16,5))
    
    # Plot Coma's Mean Recessional Velocity & Overlay with Coma Galaxies from NED
    ax2 = ax1.twiny()
    ax2.plot([arcmin_min, arcmin_max], (c*coma_z, c*coma_z),      # Mean Velocity
             'blue', linewidth=2)
    ax2.scatter(coma_dist, coma_vel,
                s=10, marker='o', c='cornflowerblue',
                linewidths=0.3, alpha=0.4) # Coma Galaxies

    # UDGs
    for idx,sep in enumerate(separation):
        ax2.scatter(sep, df["cz"].iloc[idx],
                    color=color[idx], marker=marker[idx], label=label[idx],
                    s=marker_size * df["Reff"].iloc[idx]**2,
                    alpha=1,
                    linewidths=2 if df["Reff"].iloc[idx] > large_thres else 0.2,
                    edgecolors='k')

    ax2.set_xlim([arcmin_min, arcmin_max]) #arcmin
    ax2.set_ylim([kms_min, kms_max]) #km/s
    ax2.set_xlabel("$r_\mathrm{proj} \, (\mathrm{arcmin})$", size=label_size)
    ax2.xaxis.labelpad = 20
    plt.minorticks_on()
    plt.tick_params(which='both', direction='in', pad=10, width=1.5)
    ax2.tick_params(which='major', length=5)
    ax2.tick_params(which='minor', length=2)
    ax2.xaxis.set_ticks(np.arange(0,505,100))
    ax2.yaxis.set_ticks(np.arange(2000,12005,2000))
    
    # Unique Markers in Legend Only (Uses Markers w/o Bold Outline)
    handles, labels = ax2.get_legend_handles_labels()
    handles = handles[::-1]
    labels  = labels[::-1]
    unique  = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) \
                      if l not in labels[:i]]
    legend  = ax2.legend(*zip(*unique), loc=legend_loc,
                         fancybox=True,
                         frameon=True,
                         prop={'size': 20},
                         title_fontsize=24,
                         title=legend_title)

    # Set Marker Size in Legend to `small_thres` Size
    for legend_handle in legend.legendHandles:
        legend_handle._sizes = [marker_size * small_thres**2]
    
    # Sets Axes Line Width
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(1.5)

    # Removes Border Whitespace & Save
    plt.tight_layout()
    plt.savefig(plot_fname, format='pdf')
    plt.close()


###############################################################################

def main(data_dir='../data', plot_dir='../plots', udg_only=True, local_env=True):
    
    # Appropriate File Names
    prefix = ('udgs'  if udg_only  else 'candidates') + '_' + \
             ('local' if local_env else 'global')     + '_'
    plot_fname = os.path.join(plot_dir, prefix + "phasespace.pdf")

    # NED Coma Galaxies Data
    ned_fname  = os.path.join(data_dir,'objsearch_cz2000-12000_500arcmin.txt')
    data_fname = os.path.join(data_dir,'kadowaki2019.tsv')
    

    phase_space_plot(data_fname, ned_fname, plot_fname=plot_fname,
                     udg_only=udg_only,
                     local_env=local_env)


###############################################################################

if __name__ == '__main__':
    main(udg_only=True,  local_env=True)
    main(udg_only=True,  local_env=False)
    main(udg_only=False, local_env=True)
    main(udg_only=False, local_env=False)

###############################################################################
