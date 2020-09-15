#!/usr/bin/env python

import colossus
from colossus.cosmology import cosmology
from colossus.lss import mass_function as mf
cosmology.setCosmology('planck18')

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
font = {'family' : 'normal', 'size'   : 16}
plt.rc('font', **font)

import numpy as np

#######################################################################

def get_bounds(arr):
    """
    Computes plot bounds with of nearest integer exponents.
    """
    plot_min = 10**np.ceil(  np.log10(max(arr)) )
    plot_max = 10**np.floor( np.log10(min(arr)) )

    return plot_min, plot_max


def get_number_density(m, mfunc, dm=None):
    """
    Performs integration of Halo Mass Function.
    
    Args:
        m (np.ndarray): Array of Exponent of Halo Masses in M_sun.
        mfunc (np.ndarray): Array of Corresponding Halo Mass Functions
        dm (float): Exponent of Mass Resolution for integration.
                    [i.e., default dM = 10^7 to integrate dn/dln(M)]
    """

    if dm == None:
        # Assumes Linearly-Spaced Data
        dm = m[1] - m[0]
        
    return sum(mfunc/m) * 10**dm

#######################################################################

def fraction_of_massfunction(halo_015=10.8, halo_060=12.1, halo_100=12.5,
                             z=0.0231, delta_mass=7,
                             num_pts=100, plot_fname=None):
    """
    Args:
        halo_015 (float): Exponent of Halo Mass for a  1.5 kpc UDG in M_sun.
        halo_060 (float): Exponent of Halo Mass for a  6.0 kpc UDG in M_sun.
        halo_100 (float): Exponent of Halo Mass for a 10.0 kpc UDG in M_sun.
        z (float): Redshift
        delta_mass (float): Exponent of Mass Resolution for integration.
                            [i.e., default delta_mass = 10^7 to integrate dn/dln(M)]
        num_pts (int): Number of Points to Plot.
        plot_fname (str): Filename. If set to None, plot is not generated.
    
    """

    print("\n---------------------------------------")
    print("\nPARAMETERS")
    print("1.5  kpc Halo Mass: 10^{0} M_sun".format(halo_015))
    print("6.0  kpc Halo Mass: 10^{0} M_sun".format(halo_060))
    print("10.0 kpc Halo Mass: 10^{0} M_sun".format(halo_100))
    print("\n---------------------------------------")

    # Compute Mass Function (Linearly-spaced Data)
    M     = np.arange(10**halo_015, 10**halo_100, 10**delta_mass)
    mfunc = mf.massFunction(M, z, mdef='200m',
                            model='tinker08', q_out='dndlnM')

    # Integration
    M_over6     = M[np.where(M >= 10**halo_060)]
    mfunc_over6 = mfunc[np.where(M >= 10**halo_060)]
    over6_sum   = get_number_density(M_over6, mfunc_over6, dm=delta_mass)
    total       = get_number_density(M,       mfunc,       dm=delta_mass)

    print("\nSamples of (r_e >= 6.0 kpc) Galaxies:  ", len(M_over6))
    print("Sum for Galaxies with (r_e >= 6.0 kpc):",   np.round(over6_sum,4))
    print("Total Samples of All Galaxies:         ",   len(M))
    print("Total for all Galaxies:                ",   np.round(total,4))
    print("---------------------------------------")
    print("Fraction:", np.round(over6_sum/total,4), "\n")


    if plot_fname:
        
        # Logarithmically-spaced Data (ideal for plotting on log-scale.)
        M     = np.logspace(halo_015, halo_100, num_pts)
        mfunc = mf.massFunction(M, z, mdef='200m',
                                model='tinker08', q_out='dndlnM')

        # Plot Mass Function
        plt.clf()
        plt.figure()
        plt.plot(M, mfunc, '-', c='b', label=r'Mass Function')
        plt.plot([10**halo_015]*2, [min(mfunc), max(mfunc)],
                 c='m', linestyle='--', label=r'$1.5 \, \mathrm{kpc}$')
        plt.plot([10**halo_060]*2, [min(mfunc), max(mfunc)],
                 c='r', linestyle='--', label=r'$6.0 \, \mathrm{kpc}$')
        plt.plot([10**halo_100]*2, [min(mfunc), max(mfunc)],
                 c='g', linestyle='--', label=r'$10.0 \, \mathrm{kpc}$')
        
        # Customize Plot
        plt.ylim(get_bounds(mfunc))
        plt.loglog()
        plt.xlabel(r'$M_{200} \, (\mathrm{M}_\odot)$')
        plt.ylabel(r'$dn/d\ln(M)$')
        plt.legend(loc='best')
        
        # Save Figure
        plt.savefig(plot_fname, bbox_inches='tight')
        plt.close()

#######################################################################

if __name__ == "__main__":

    fraction_of_massfunction(halo_015=10.8,
                             halo_060=12.0,
                             halo_100=12.5,
                             z=0.0231,
                             delta_mass=7,
                             num_pts=100,
                             plot_fname='../plots/mass_function.pdf')
￼
