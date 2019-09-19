#! /usr/bin/env python

################################################################################

from conversions import get_angular_size, get_physical_size
import numpy as np
import matplotlib.pyplot as plt


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

def load_OUR_data(fname):
    """
    """
    # Our Spectroscopic Survey Data
    data   = np.genfromtxt(fname, dtype=None, encoding='ascii')
    ra, dec, z, env, size = list(zip(*data))
    velocity   = c * np.array(z)
    separation = get_angular_size(ra, dec, coma_ra, coma_dec)/60 # arcmin
    
    return separation, velocity, size, env

###############################################################################

def phase_space_plot(data_fname,
                     ned_fname, plot_fname, local_env=True,
                     kms_limit=[200,12050],
                     arcmin_limit=[0,505],
                     mpc_limit=[0,14.76],
                     adjust_marker=True, legend_loc='lower right'):

    """
    Creates phase space diagram
    
    ARGS:
    
    RETURNS:
    
    """
    
    coma_vel, coma_dist = load_NED_data(ned_fname)
    separation, velocity, size, env = load_OUR_data(data_fname)

    # Select Legend Labels, Marker Sizes & Colors & Shapes
    marker_size = 40  # Size of Marker for R_e = 1.0 kpc
    large_thres = 3.5 # kpc
    
    if local_env:
        label = [ r'$\mathrm{Dense}$'  if val=='Dense'  else \
                  r'$\mathrm{Sparse}$' if val=='Sparse' else \
                  r'$\mathrm{Unconstrained}$' for val in env]

        color = [ 'lime'   if val=='Dense'  else \
                  'orange' if val=='Sparse' else \
                  'blue' for val in env]

        marker = [ '^' if val=='Dense'  else \
                   'o' if val=='Sparse' else \
                   'x' for val in env]
    else:
        label = [ r'$\mathrm{Cluster}$'                if val=='Cluster'     else \
                  r'$\mathrm{Non}$-$\mathrm{Cluster}$' if val=='Non-Cluster' else \
                  r'$\mathrm{Unconstrained}$' for val in env]
          
        color = [ 'lime'   if val=='Cluster'      else \
                  'orange' if val=='Non-Cluster' else \
                  'blue'   for val in env]

        marker = [ '^' if val=='Cluster'     else \
                   'o' if val=='Non-Cluster' else \
                   'x' for val in env]
    
    # Create Figure
    plt.clf()
    plt.rcParams['savefig.facecolor'] = "1."
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size='20')

    # Establish Axis Limits & Labels
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax1.set_xlim(mpc_limit)
    ax1.set_xlabel("$\mathrm{Projected \, Distance} \, (\mathrm{Mpc})$", size=20)
    ax1.set_ylabel("$cz \, (\mathrm{km \, sec^{-1}})$", size=24)
    
    # Plot Splashback Radius
    ax1.plot((r_splash,r_splash), kms_limit, 'r--', linewidth=3)

    # Plot Coma's Mean Recessional Velocity & Overlay with Coma Galaxies from NED
    ax2 = ax1.twiny()
    ax2.plot(arcmin_limit, (c*coma_z, c*coma_z), 'b', linewidth=2)   # Mean Velocity
    ax2.scatter(coma_dist, coma_vel, s=1, marker='.')                # Coma Galaxies

    # UDGs
    for idx,sep in enumerate(separation):
        ax2.scatter(sep, velocity[idx],
                    c=color[idx], label=label[idx], marker=marker[idx],
                    s=60*size[idx] if adjust_marker else 60,
                    alpha=1,
                    linewidths=2 if size[idx] > large_thres else 0.2,
                    edgecolors='k')

    ax2.set_xlim(arcmin_limit) #arcmin
    ax2.set_ylim(kms_limit) #km/s
    ax2.set_xlabel("$r \, (\mathrm{arcmin})$", size=24)

    handles, labels = ax2.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    legend = ax2.legend(*zip(*unique), fancybox=True, prop={'size': 10},
                        loc=legend_loc, frameon=True, title_fontsize=12,
                        title=r'$\mathrm{Local \, Environment}$' if local_env
                              else r'$\mathrm{Coma \, Membership}$')

    for legend_handle in legend.legendHandles:
        legend_handle._sizes = [marker_size * 1.5]

    plt.tight_layout()
    plt.savefig(plot_fname, format='pdf')


###############################################################################

def main(udgs_only=True, local_env=True):
    
    # Appropriate File Names
    if local_env:
        data_fname = '../data/redshifts2_local_udgs.dat'   if udgs_only else \
                     '../data/redshifts2_local_candidates.dat'
        plot_fname = '../plots/phasespace_local_udgs.pdf'  if udgs_only else \
                     '../plots/phasespace_local_candidates.pdf'
    else:
        data_fname = '../data/redshifts2_global_udgs.dat'  if udgs_only else \
                     '../data/redshifts2_global_candidates.dat'
        plot_fname = '../plots/phasespace_global_udgs.pdf' if udgs_only else \
                     '../plots/phasespace_global_candidates.pdf'

    # NED Coma Galaxies Data
    ned_fname = '../data/objsearch_cz2000-12000_500arcmin.txt'
    
    # Plot Limits
    kms_min    = 2000  if udgs_only else 0
    kms_max    = 12050 if udgs_only else 13000
    arcmin_min = 0
    arcmin_max = 505   if udgs_only else 650
    mpc_min    = 0
    mpc_max    = get_physical_size(60*arcmin_max, c*coma_z, H0=h0)
    
    # Plot Phase Space Data
    legend_loc = 'lower right' if udgs_only else 'upper right'

    phase_space_plot(data_fname, ned_fname,  plot_fname, local_env=local_env,
                     kms_limit    = [kms_min,    kms_max],
                     arcmin_limit = [arcmin_min, arcmin_max],
                     mpc_limit    = [mpc_min,    mpc_max],
                     adjust_marker=True,
                     legend_loc=legend_loc)


###############################################################################

if __name__ == '__main__':
    main(udgs_only=True,  local_env=True)
    main(udgs_only=True,  local_env=False)
    main(udgs_only=False, local_env=True)
    main(udgs_only=False, local_env=False)

###############################################################################
