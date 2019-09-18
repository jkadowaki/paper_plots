#!/usr/bin/env python

import colossus
from colossus.lss import peaks
from colossus.halo import splashback
from colossus.cosmology import cosmology
import numpy as np

# Set Cosmology
cosmology.setCosmology('planck18')
c          = 299792         # km/s (speed of light)
rad_to_sec = 206265         # arcseconds in a radian
mpc_to_kpc = 1000           # kpc in a Mpc
min_to_sec = 60             # arcseconds in a arcminute
H0         = 67.37          # (km/s)/Mpc (Hubble's Constant)
h          = H0/100         # Planck18


# COMA PARAMETERS
# r_vir: https://academic.oup.com/mnras/article/343/2/401/1038976
# R200, M200: https://iopscience.iop.org/article/10.1086/523101/pdf
r_vir      = 2.6              # Mpc
R200m      = 1.99/h           # Mpc; Kubo 2007 (Virial Radius Estimate)
M200m      = 1.88/h * 10**15  # Kubo 2007
z          = 0.0231
x          = (c * z * mpc_to_kpc * min_to_sec) / (H0 * rad_to_sec) # kpc in arcmin at Coma Distance

# COLOSSUS CALCULATION OF SPLASHBACK RADIUS
# PAPER:   https://arxiv.org/pdf/1703.09716.pdf
# PACKAGE: https://bdiemer.bitbucket.io/colossus/halo_splashback.html
nu200m         = peaks.peakHeight(M200m, z)
RspR200m, mask = splashback.splashbackModel('RspR200m', nu200m = nu200m, z = z)
splash         = RspR200m * R200m

# Splashback Radius (in units of R200)
print("\nVirial Radius:    ", r_vir)
print("R200 Radius:      ",   np.round(R200m,3))
print("Splashback Radius:",   np.round(RspR200m * R200m,3))  # Mpc
print("r_splash / r_200: ",   np.round(RspR200m,3), "\n")    # R200
