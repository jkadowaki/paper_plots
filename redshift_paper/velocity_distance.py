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
udgs_only = True

def arcmin_to_mpc(arcminute, redshift):

    # Projected Distance at redshift z
    # For local galaxies, v = cz = H0 d --> d = cz/H0
    # Use angular diameter distance otherwise: comoving = c/H0 * int dz/sqrt(OL + OM * (1+z)^3) from 0 to z
    arcsec_in_arcmin = 60.
    arcsec_in_radian = 206265.
    
    theta = float(arcminute) * arcsec_in_arcmin / arcsec_in_radian
    comoving = c * float(redshift) / h0
    
    return comoving * theta / (1.+redshift)


# SURVEY GALAXIES
fname           = 'redshifts2_udgs.dat' if udgs_only else 'redshifts2_candidates.dat'
data            = np.genfromtxt(fname, dtype=None)
ra, dec, z, env = list(zip(*data))
velocity        = [c*redshift for redshift in z]
ra_offset       = (np.array(ra)-coma_ra) * np.cos((np.array(dec)+coma_dec)/2 * np.pi/180.)
dec_offset      = np.array(dec)-coma_dec
separate        = np.sqrt( (ra_offset*60)**2 + (dec_offset*60)**2 )

label = [     '$\mathrm{Dense}$'  if val==b'Dense'
         else '$\mathrm{Sparse}$' if val==b'Sparse'
         else '$\mathrm{Unconstrained}$' for val in env]

color = ['b' if val==b'Dense'
         else 'r' if val==b'Sparse'
         else 'g' for val in env]

marker = ['^' if val==b'Dense'
          else 'o' if val==b'Sparse'
          else 'x' for val in env]

# COMA CLUSTER GALAXIES
fname	  = 'objsearch_cz2000-12000_500arcmin.txt'
coma      = np.genfromtxt(fname, dtype=None, skip_header=24, delimiter='\t')
coma_vel  = [coma_galaxy[5] for coma_galaxy in coma]
coma_dist = [coma_galaxy[9] for coma_galaxy in coma]


arcmin_min = 0
arcmin_max = 505 if udgs_only else 650
mpc_min    = 0
mpc_max    = arcmin_to_mpc(arcmin_max, z_coma)
kms_min    = 2000 if udgs_only else 0
kms_max    = 12050 if udgs_only else 13000


plt.clf()
plt.rcParams['savefig.facecolor'] = "1."
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size='20')

fig, ax1 = plt.subplots(figsize=(10,8))
ax1.set_xlim(mpc_min, mpc_max)
ax1.set_xlabel("$\mathrm{Projected \, Distance} \, (\mathrm{Mpc})$", size=20)
ax1.set_ylabel("$cz \, (\mathrm{km \, sec^{-1}})$", size=24)
ax1.plot((r200,r200),(kms_min, kms_max), 'y--', linewidth=3)   # Virial Radius

ax2 = ax1.twiny()
ax2.plot((arcmin_min,arcmin_max),(c*z_coma, c*z_coma), 'g', linewidth=2)   # Coma Cluster
ax2.scatter(coma_dist, coma_vel, s=1, marker='.')                          # Coma Galaxies

# Write out separation-velocity data
#data_fname = "velocity_distance_udgs.txt" if udgs_only else "velocity_distance_candidates.txt"
#f = open(data_fname, 'w+')

# UDGs
for idx,sep_val in enumerate(separate):
    ax2.scatter(sep_val, velocity[idx],
                c=color[idx], label=label[idx], marker=marker[idx],
                s=40, linewidths=0.3, alpha=1, edgecolors='k')

    
    #f.write("{0}\t{1}\t{2}\n".format(name[idx],
    #                                 arcmin_to_mpc(sep_val, z[idx]),
    #                                 velocity[idx]))

ax2.set_xlim(arcmin_min,arcmin_max) #arcmin
ax2.set_ylim(kms_min, kms_max) #km/s
ax2.set_xlabel("$r \, (\mathrm{arcmin})$", size=24)

handles, labels = ax2.get_legend_handles_labels()
unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
ax2.legend(*zip(*unique),  fancybox=True, prop={'size': 10}, #ncol=3,
           loc='lower right' if udgs_only else 'upper right', #bbox_to_anchor=(0.02, 0.125),
           title=r'$\mathrm{Environment}$', title_fontsize=15)

plt.tight_layout()

plot_fname = 'phase_space_udgs.pdf' if udgs_only else 'phase_space_candidates.pdf'
plt.savefig(plot_fname, format='pdf')

#f.close()
