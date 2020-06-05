#!/usr/bin/env python3

################################################################################

from __future__ import print_function
from math import log10
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from math import log10, sqrt

from astropy import units as u
from astropy.coordinates import SkyCoord
from astroML.plotting import setup_text_plots

setup_text_plots(fontsize=20, usetex=True)


### CONSTANTS ###
omega_m   = 0.3
omega_lam = 0.7
H0        = 70.         # Hubble parameter at z=0, km/s/Mpc
c_kms     = 299792.458  # speed of light, km/s
dH        = c_kms / H0  # Hubble distance, Mpc


################################################################################

def inv_efunc(z):
    """ Used to calculate the comoving distance to object at redshift
        xs    z. Eqn 14 from Hogg, astro-ph/9905116."""
    return 1. / sqrt(omega_m * (1. + z)**3 + omega_lam)


################################################################################

def sizehist( data_fname='coma_vicinity.dat',
              udg_size_fname='re.dat',
              plot_fname='sizehist_final.pdf' ):

    #############################   SDSS GALAXIES   ############################ 
   
    # Load galaxy positions, magnitudes and redshifts and r-i colours.
    r = np.genfromtxt(data_fname, dtype=None, skip_header=2,
                      names='ra,dec,cz,g,dm1,re',
                      usecols=(0, 1, 2, 3, 5, 6))
    # Apply Redshift & Size Cuts
    condition1 = (r['cz']>0.015) & (r['cz']<0.0333) & (r['re']>0)
    r = r[condition1]
    #Calculate the comoving distance corresponding to each object's redshift       
    dist = np.array([dH * integrate.quad(inv_efunc, 0, cz)[0] for cz in r['cz']])


    #################################   UDGS   #################################
   
    t = np.genfromtxt(udg_size_fname, dtype=None, skip_header=0,
                      names='re',usecols=(0))
    dm   = 5*log10(98.0*1020000.0) - 5
    rlen = len(r['g'])

    for i in range(0,rlen):
        r['dm1'][i] = (5*log10(dist[i]*1020000.0) - 5)

    r['g']  = r['g'] - r['dm1']
    r['re'] = (r['re']*1000*dist/206265.0)
 
    condition3 = (r['g']<-17) & (r['g']>-20) & (r['re']>2.4) & (r['re']<10)
    r = r[condition3]


    ############################   SIZE HISTOGRAM   ############################
    
    # Create Figure
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(bottom=0.2, top=0.9,
                        left=0.2, right=0.9, wspace=None, hspace = None)
    # SDSS Galaxies
    n, bins, patches = plt.hist(r['re'], 25, density=1, histtype='stepfilled',
                                facecolor='darkseagreen', alpha=0.45)
    # UDGs
    n, bins, patches = plt.hist(t['re'],10, density=1, facecolor='r', alpha=0.3)
                               # facecolor='darkslategrey', alpha=0.45)
    # Plot Formatting
    plt.tick_params(which='both', direction='in', pad =10, labelsize=20)
    plt.yscale('log')
    plt.ylabel(r'log(Normalized Number)')
    plt.xlabel(r'$r_e$ (kpc)')
    plt.axis([2.4,10,0.001,3])

    # Save Figure
    plt.tight_layout()
    plt.savefig(plot_fname, bbox_inches='tight')
    plt.show()


################################################################################

if __name__=="__main__":

    sizehist( data_fname='../data/coma_vicinity.dat', 
              udg_size_fname='../data/re.dat',
              plot_fname='../plots/sizehist_final.pdf' )

################################################################################
