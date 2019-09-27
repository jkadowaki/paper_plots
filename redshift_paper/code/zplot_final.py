#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from math import sqrt
import os
import pandas as pd
from pair_plot import read_data, get_label_color_marker
from scipy import integrate


# CONSTANTS
# To plot the space distribution we need to convert redshift to
# distance.  The values and function below are needed for this
# conversion.
omega_m = 0.3147
omega_lam = 0.6853
H0 = 70.            # Hubble parameter at z=0, km/s/Mpc
c_kms = 299792.458  # speed of light, km/s
dH = c_kms / H0     # Hubble distance, Mpc

# Coma Parameters
coma_ra = 194.952917

fontsize = 20


###############################################################################

def inv_efunc(z):
    """ Used to calculate the comoving distance to object at redshift
    z. Eqn 14 from Hogg, astro-ph/9905116."""
    return 1. / sqrt(omega_m * (1. + z)**3 + omega_lam)

###############################################################################

def zplot(redshift_file='../data/kadowaki2019.tsv',
          zplot_file='zplot.pdf',
          plot_dir='../plots',
          mfeat="Reff",
          udg_only=True,
          local_env=True):

    # Define Environment
    efeat = 'LocalEnv' if local_env else 'GlobalEnv'

    # Now read the LRG positions, magnitudes and redshifts and r-i colours.
    # Coma Galaxies
    r = np.genfromtxt('../data/coma_vicinity.dat', dtype=None, skip_header=2,
                      names='ra,dec,redshift,r,g,x0,y0',
                      usecols=(0, 1, 2, 3, 4, 5, 5))
    # UDGs
    q = read_data(redshift_file, udg_only=udg_only, field='Coma')
    q = q[["ra", "dec", "redshift", mfeat, efeat]].dropna()
    q = q.sort_values(by=[mfeat], ascending=False)
    
    # Calculate the comoving distance corresponding to each object's redshift
    dist  = np.array([dH * integrate.quad(inv_efunc, 0, z)[0] for z in r['redshift']])
    distq = np.array([dH * integrate.quad(inv_efunc, 0, z)[0] for z in q['redshift']])

    
    # Plot the distribution of galaxies, converting redshifts to positions
    # assuming Hubble flow.
    # Coma Galaxies
    theta   = (r['ra']-coma_ra) * np.pi / 180  # radians
    r['y0'] =  dist * np.cos(theta)
    r['x0'] = -dist * np.sin(theta)
    # UDGs
    thetaq  = (q['ra']-coma_ra) * np.pi / 180  # radians
    q['y0'] =  distq * np.cos(thetaq)
    q['x0'] = -distq * np.sin(thetaq)
    
    # Coma Galaxies
    condition  = (r['redshift'] > 0.0) & (r['redshift']<0.041) & (abs(theta)  < 0.20)
    r = r[condition]
    # UDGs
    conditionq = (q['redshift'] > 0.0) & (q['redshift']<0.041) & (abs(thetaq) < 0.20)
    q = q[conditionq]

    label, color, marker, legend_title = get_label_color_marker(q, efeat)
    marker_edge = 'k'
    thin_line   = 0.2
    thick_line  = 1.5
    
    for idx in range(len(q)):
        print(idx, q['ra'].iloc[idx], q['dec'].iloc[idx],
              label[idx], color[idx], marker[idx])
    

    # Make the area of each circle representing a galaxy position
    # proportional to its apparent r-band luminosity.
    #sizes = 30 * 10**-((r['rmag'] - np.median(r['rmag']))/ 2.5)
    sizes = 10
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.35, right=0.7, wspace=None, hspace = None)

    # Plot the galaxies, colouring points by z.
    col = plt.scatter(r['x0'], r['y0'], marker='.', s=sizes, c='cornflowerblue', linewidths=0.3,alpha=0.4)

    # UDGs
    sizesq = 20       # Size of Marker for R_e = 1.0 kpc
    large_thres = 3.5 # kpc
    for idx in range(len(q)):
        col = plt.scatter(q['x0'].iloc[idx], q['y0'].iloc[idx],
                          label  = label[idx],
                          color  = color[idx],
                          marker = marker[idx],
                          s      = sizesq * (q[mfeat].iloc[idx])**2,
                          alpha  = 1,
                          edgecolors=marker_edge,
                          linewidth = thick_line if q[mfeat].iloc[idx]>large_thres else thin_line)


    plt.xlabel('$\mathrm{X \, (Mpc)}$',        fontsize=fontsize)
    plt.ylabel('$\mathrm{Distance \, (Mpc)}$', fontsize=fontsize)

    plt.axis([-40,40,-10,180])
    plt.tick_params(which='both', direction='in', pad=10, labelsize=fontsize)
    ax.xaxis.set_ticks(np.arange(-35,35,5),minor=True)
    ax.yaxis.set_ticks(np.arange(-5,175,5),minor=True)

    handles, labels = ax.get_legend_handles_labels()
    handles = handles[::-1]
    labels  = labels[::-1]
    unique  = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    legend = ax.legend(*zip(*unique), fancybox=True, prop={'size': 15},
                       loc='lower right',  frameon=True, title_fontsize=18,
                       title=legend_title)

    for legend_handle in legend.legendHandles:
        legend_handle._sizes = [sizesq * 1.5**2]  # Marker Size in Legend is Threshold UDG Size

    prefix = ('udgs'  if udg_only  else 'candidates') + '_' + \
             ('local' if local_env else 'global')     + '_'
    plt.savefig(os.path.join(plot_dir, prefix + zplot_file), bbox_inches='tight')
    plt.close()

    
###############################################################################

if __name__ == '__main__':

    zplot(udg_only=False, local_env=False)
    zplot(udg_only=False, local_env=True)
    zplot(udg_only=True,  local_env=False)
    zplot(udg_only=True,  local_env=True)
   
