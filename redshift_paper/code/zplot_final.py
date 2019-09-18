#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from scipy import integrate
from math import sqrt

fontsize = 20
#from astroML.plotting import setup_text_plots
#setup_text_plots(fontsize=20, usetex=True)

# To plot the space distribution we need to convert redshift to
# distance.  The values and function below are needed for this
# conversion.
omega_m = 0.3
omega_lam = 0.7
H0 = 70.    # Hubble parameter at z=0, km/s/Mpc
c_kms = 299792.458 # speed of light, km/s
dH = c_kms / H0          # Hubble distance, Mpc
udg_only = True

def inv_efunc(z):
    """ Used to calculate the comoving distance to object at redshift
    z. Eqn 14 from Hogg, astro-ph/9905116."""
    return 1. / sqrt(omega_m * (1. + z)**3 + omega_lam)

###############################################################################

def zplot(redshift_file, zplot_file, udgs_only=True, local_env=True):


    # Now read the LRG positions, magnitudes and redshifts and r-i colours.
    #r = np.genfromtxt('coma_sdss.dat', dtype=None, skip_header=26,
    #                  names='RA,Dec,z',
    #                  usecols=(3, 4, 7))
    r = np.genfromtxt('../data/coma_vicinity.dat', dtype=None, skip_header=2,
                      names='ra,dec,z,r,g,x0,y0',
                      usecols=(0, 1, 2, 3, 4, 5, 5))

    q = np.genfromtxt(redshift_file, dtype=None, skip_header=0,
                      names='ra,dec,z,env,re,x0,y0',
                      usecols=(0, 1, 2, 3, 4, 2, 2))


    # Calculate the comoving distance corresponding to each object's redshift
    dist = np.array([dH * integrate.quad(inv_efunc, 0, z)[0] for z in r['z']])
    distq = np.array([dH * integrate.quad(inv_efunc, 0, z)[0] for z in q['z']])


    # Plot the distribution of galaxies, converting redshifts to positions
    # assuming Hubble flow.
    theta = (r['ra']-194.952917) * np.pi / 180  # radians
    r['y0'] = dist * np.cos(theta)
    r['x0'] = -dist * np.sin(theta)

    thetaq = (q['ra']-194.952917) * np.pi / 180  # radians
    q['y0'] = distq * np.cos(thetaq)
    q['x0'] = -distq * np.sin(thetaq)

    condition = (r['z'] > 0.0) & (r['z']<0.041) & (abs(theta) < 0.20)
    r = r[condition]

    conditionq = (q['z'] > 0.0) & (q['z']<0.041) & (abs(thetaq) < 0.20)
    q = q[conditionq]

    if local_env:
        label  = {b'Dense':'$\mathrm{Dense}$', b'Sparse':'$\mathrm{Sparse}$'}  # Default: '$\mathrm{Unconstrained}$'
        color  = {b'Dense':'lime',             b'Sparse':'orange'}             # Default: Use 'g'
        marker = {b'Dense':'^',                b'Sparse':'o'}                  # Default: Use 'x'
    else:
        label  = {b'Cluster':'$\mathrm{Cluster}$',   b'Non-Cluster':'$\mathrm{Non}$-$\mathrm{Cluster}$'}  # Default: '$\mathrm{Unconstrained}$'
        color  = {b'Cluster':'lime',               b'Non-Cluster':'orange'}                             # Default: Use 'g'
        marker = {b'Cluster':'^',                  b'Non-Cluster':'o'}                                  # Default: Use 'x'


    for idx in range(q.size):
        print(idx, q['ra'][idx], q['dec'][idx],
              label.get( q['env'][idx], '$\mathrm{Unconstrained}$'),
              color.get( q['env'][idx], 'b'),
              marker.get(q['env'][idx], 'x'))
    

    # Make the area of each circle representing a galaxy position
    # proportional to its apparent r-band luminosity.
    #sizes = 30 * 10**-((r['rmag'] - np.median(r['rmag']))/ 2.5)
    sizes = 10
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.35, right=0.7, wspace=None, hspace = None)

    # Plot the galaxies, colouring points by z.
    col = plt.scatter(r['x0'], r['y0'], marker='.', s=sizes, c='lightsteelblue', linewidths=0.3,alpha=0.4)

    sizesq = 20       # Size of Marker for R_e = 1.0 kpc
    large_thres = 3.5 # kpc
    for idx in range(q.size):
        col = plt.scatter(q['x0'][idx], q['y0'][idx],
                          label  = label.get( q['env'][idx], '$\mathrm{Unconstrained}$'),
                          color  = color.get( q['env'][idx], 'b'),
                          marker = marker.get(q['env'][idx], 'x'),
                          s      = sizesq * q['re'][idx],
                          linewidth = 1.5 if q['re'][idx]>large_thres else 0.2,
                          alpha=1,
                          edgecolors='k')


    plt.xlabel('$\mathrm{X \, (Mpc)}$',        fontsize=fontsize)
    plt.ylabel('$\mathrm{Distance \, (Mpc)}$', fontsize=fontsize)

    plt.axis([-40,40,-10,180])
    plt.tick_params(which='both', direction='in', pad=10, labelsize=fontsize)
    ax.xaxis.set_ticks(np.arange(-35,35,5),minor=True)
    ax.yaxis.set_ticks(np.arange(-5,175,5),minor=True)

    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    legend = ax.legend(*zip(*unique), fancybox=True, prop={'size': 10},
                       loc='lower right',  frameon=True, title_fontsize=12,
                       title=r'$\mathrm{Local \, Environment}$' if local_env else r'$\mathrm{Coma \, Membership}$')

    for legend_handle in legend.legendHandles:
        legend_handle._sizes = [sizesq * 1.5]  # Marker Size in Legend is Threshold UDG Size

    plt.savefig(zplot_file, bbox_inches='tight')

###############################################################################

def main():
    
    zplot('../data/redshifts2_local_candidates.dat',
          '../plots/zplot_local_candidates.pdf',
          udgs_only=False, local_env=True)
    zplot('../data/redshifts2_global_candidates.dat',
          '../plots/zplot_global_candidates.pdf',
          udgs_only=False, local_env=False)
    zplot('../data/redshifts2_local_udgs.dat',
          '../plots/zplot_local_udgs.pdf',
          udgs_only=True,  local_env=True)
    zplot('../data/redshifts2_global_udgs.dat',
          '../plots/zplot_global_udgs.pdf',
          udgs_only=True,  local_env=False)

###############################################################################

if __name__ == '__main__':
    main()
