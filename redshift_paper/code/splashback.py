#!/usr/bin/env python

from __future__ import print_function
import colossus
from colossus.lss import peaks
from colossus.halo import splashback
from colossus.cosmology import cosmology
import numpy as np

# Set Cosmology & Constants
cosmology.setCosmology('planck18')
H0      = 67.37         # Hubble's Constant [(km/s) / Mpc]  (Planck 2018)
c       = 299792        # Speed of Light (km/s)
RAD2SEC = 206265        # Arcseconds in a Radian
MIN2SEC = 60            # (Arc)Seconds in a(n) (Arc)Minute
MPC2KPC = 1000          # kpc in a Mpc

NUM_DECIMAL_PLACES = 3  # User-defined rounding preference


################################################################################
"""
SPLASHBACK.py
    Computes the splashback radius of a galaxy cluster using COLOSSUS.
    
    # COLOSSUS CALCULATION OF SPLASHBACK RADIUS
    # PAPER:   https://arxiv.org/pdf/1703.09716.pdf
    # PACKAGE: https://bdiemer.bitbucket.io/colossus/halo_splashback.html
    
    Methods:
    (1) GET_SPLASHBACK_RADIUS: Computes splashback radius given enclosed mass.
"""
################################################################################

def get_splashback_radius(R200m, M200m, z, r_virial=None, verbose=True):
    """
    Computes the splashback radius of a galaxy cluster.
    
    ARGS:
        R200m (float): r_200 Radius of Galaxy Cluster
                       (Central density is 200x denser than density at r_200.)
        M200m (float): Enclosed mass of the galaxy cluster (M_sun) in r_200.
        z (float): Redshift of Galaxy Cluster
        r_vir (float): Virial Radius of Galaxy Cluster
        
    RETURNS:
        splash (float): The splashback radius (Mpc)
    """
    if verbose and r_virial:
        print("\nVirial Radius:    ", r_virial)
    
    # Peak Height
    nu200m = peaks.peakHeight(M200m, z)
    
    # Splashback Radius (in units of R200)
    RspR200m, mask = splashback.splashbackModel('RspR200m', nu200m=nu200m, z=z)
    
    # Splashback Radius (in Mpc)
    splash = RspR200m * R200m

    if verbose:
        print("R200 Radius:      ", np.round(R200m,    NUM_DECIMAL_PLACES))
        print("Splashback Radius:", np.round(splash,   NUM_DECIMAL_PLACES))
        print("r_splash / r_200: ", np.round(RspR200m, NUM_DECIMAL_PLACES))
        print("\n")
    
    return np.round(splash, NUM_DECIMAL_PLACES)


################################################################################

if __name__ == "__main__":

    # COMA PARAMETERS
    # r_vir: https://academic.oup.com/mnras/article/343/2/401/1038976
    # R200, M200: https://iopscience.iop.org/article/10.1086/523101/pdf
    r_vir  = 2.6                       # Mpc
    R200m  = 1.99 / (H0/100)           # Mpc; Kubo 2007 (Virial Radius Estimate)
    M200m  = 1.88 / (H0/100) * 10**15  # Kubo 2007
    z      = 0.0231

    print("\n______COMA CLUSTER______")
    splash = get_splashback_radius(R200m, M200m, z, r_vir)


################################################################################
