#! /usr/bin/env python

################################################################################

import numpy as np
import matplotlib.pyplot as plt

################################################################################

"""
    NED SEARCH RESULTS
    
    0	No.
    1	Object Name
    2	RA(deg)
    3	DEC(deg)
    4	Type
    5	Velocity
    6	Redshift
    7	Redshift Flag
    8	Magnitude and Filter
    9	Distance (arcmin)
    10	References
    11	Notes
    12	Photometry Points
    13	Positions
    14	Redshift Points
    15	Diameter Points
    16	Associations
"""

# CONSTANTS
global z_coma, c, h0
z_coma   = 0.0231
c        = 299792    # km/s
h0       = 67.37     # km/s/Mpc (Planck)
r200     = 199/h0    # Mpc
coma_ra  = 194.953
coma_dec = 27.981

###############################################################################

def arcmin_to_mpc(arcminute, redshift):

    # Projected Distance at redshift z
    # For local galaxies, v = cz = H0 d --> d = cz/H0
    # Use angular diameter distance otherwise: comoving = c/H0 * int dz/sqrt(OL + OM * (1+z)^3) from 0 to z
    arcsec_in_arcmin = 60.
    arcsec_in_radian = 206265.
    
    theta = float(arcminute) * arcsec_in_arcmin / arcsec_in_radian
    comoving = c * float(redshift) / h0
    
    return comoving * theta / (1.+redshift)


###############################################################################

def phase_space_plot(data, coma, plot_fname, local_env=True,
                     kms_limit=[200,12050], arcmin_limit=[0,505], mpc_limit=[0,14.76],
                     adjust_marker=True, legend_loc='lower right'):

    # SPECTROSCOPIC SURVEY GALAXIES
    ra, dec, z, env, size = list(zip(*data))
    velocity   = [c*redshift for redshift in z]
    ra_offset  = (np.array(ra)-coma_ra) * np.cos((np.array(dec)+coma_dec)/2 * np.pi/180.)
    dec_offset = np.array(dec)-coma_dec
    separate   = np.sqrt( (ra_offset*60)**2 + (dec_offset*60)**2 )

    if local_env:
        label = [     '$\mathrm{Dense}$'  if val==b'Dense'
                 else '$\mathrm{Sparse}$' if val==b'Sparse'
                 else '$\mathrm{Unconstrained}$' for val in env]

        color = [     'green' if val==b'Dense'
                 else 'orange' if val==b'Sparse'
                 else 'blue' for val in env]

        marker = [     '^' if val==b'Dense'
                  else 'o' if val==b'Sparse'
                  else 'x' for val in env]
    else:
        label = [     '$\mathrm{Member}$'                if val==b'Member'
                 else '$\mathrm{Non}$-$\mathrm{Member}$' if val==b'Non-Member'
                 else '$\mathrm{Unconstrained}$' for val in env]
            
        color = [     'green'  if val==b'Member'
                 else 'orange' if val==b'Non-Member'
                 else 'blue'   for val in env]

        marker = [     '^' if val==b'Member'
                  else 'o' if val==b'Non-Member'
                  else 'x' for val in env]
    
    marker_size = 40  # Size of Marker for R_e = 1.0 kpc
    large_thres = 3.5 # kpc

    # NED GALAXIES
    coma_vel  = [coma_galaxy[5] for coma_galaxy in coma]
    coma_dist = [coma_galaxy[9] for coma_galaxy in coma]
    
    plt.clf()
    plt.rcParams['savefig.facecolor'] = "1."
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size='20')

    fig, ax1 = plt.subplots(figsize=(10,8))
    ax1.set_xlim(mpc_limit)
    ax1.set_xlabel("$\mathrm{Projected \, Distance} \, (\mathrm{Mpc})$", size=20)
    ax1.set_ylabel("$cz \, (\mathrm{km \, sec^{-1}})$", size=24)
    ax1.plot((r200,r200), kms_limit, 'y--', linewidth=3)   # Virial Radius

    ax2 = ax1.twiny()
    ax2.plot(arcmin_limit, (c*z_coma, c*z_coma), 'g', linewidth=2)   # Coma Cluster
    ax2.scatter(coma_dist, coma_vel, s=1, marker='.')                          # Coma Galaxies

    # UDGs
    for idx,sep_val in enumerate(separate):
        ax2.scatter(sep_val, velocity[idx],
                    c=color[idx], label=label[idx], marker=marker[idx],
                    s=60*size[idx] if adjust_marker else 60,
                    alpha=1,
                    linewidths=1.0 if size[idx] > large_thres else 0.25,
                    edgecolors='k' if size[idx] > large_thres else 'w')

    ax2.set_xlim(arcmin_limit) #arcmin
    ax2.set_ylim(kms_limit) #km/s
    ax2.set_xlabel("$r \, (\mathrm{arcmin})$", size=24)

    handles, labels = ax2.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    legend = ax2.legend(*zip(*unique), fancybox=True, prop={'size': 10},
                        loc=legend_loc, frameon=True, title_fontsize=12,
                        title=r'$\mathrm{Local \, Environment}$' if local_env
                              else r'$\mathrm{Coma Membership}$')

    for legend_handle in legend.legendHandles:
        legend_handle._sizes = [marker_size * 1.5]

    plt.tight_layout()
    plt.savefig(plot_fname, format='pdf')


###############################################################################

def main(udgs_only=True, local_env=True):
    
    # Appropriate File Names
    if local_env:
        fname = 'redshifts2_local_udgs.dat' if udgs_only else 'redshifts2_local_candidates.dat'
        plot_fname = 'phasespace_local_udgs.pdf' if udgs_only else 'phasespace_local_candidates.pdf'
    else:
        fname = 'redshifts2_global_udgs.dat' if udgs_only else 'redshifts2_global_candidates.dat'
        plot_fname = 'phasespace_global_udgs.pdf' if udgs_only else 'phasespace_global_candidates.pdf'

    # Spectroscopic Survey Data
    data   = np.genfromtxt(fname, dtype=None)
    
    # NED Coma Galaxies Data
    fname  = 'objsearch_cz2000-12000_500arcmin.txt'
    coma   = np.genfromtxt(fname, dtype=None, skip_header=24, delimiter='\t')
    
    # Plot Limits
    kms_min    = 2000  if udgs_only else 0
    kms_max    = 12050 if udgs_only else 13000
    arcmin_min = 0
    arcmin_max = 505   if udgs_only else 650
    mpc_min    = 0
    mpc_max    = arcmin_to_mpc(arcmin_max, z_coma)
    
    # Plot Phase Space Data
    legend_loc = 'lower right' if udgs_only else 'upper right'
    
    phase_space_plot(data, coma, plot_fname, local_env=local_env,
                     kms_limit    = [kms_min,    kms_max],
                     arcmin_limit = [arcmin_min, arcmin_max],
                     mpc_limit    = [mpc_min,    mpc_max],
                     adjust_marker=True,
                     legend_loc=legend_loc)


###############################################################################

if __name__ == '__main__':
    main(udgs_only=True, local_env=True)
    main(udgs_only=True, local_env=False)
    main(udgs_only=False, local_env=True)
    main(udgs_only=False, local_env=False)

