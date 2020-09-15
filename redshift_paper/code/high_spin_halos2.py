#!/usr/bin/env python

################################################################################

import sys
sys.path.append("../code")

from read_data import read_data
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Set matplotlib to use LaTeX
font = {'size':   20}
plt.rc('text', usetex=True)
plt.rc('font', **font)

################################################################################

def best_fit(x, m, b):
    return m * x + b

################################################################################

def mag_vs_re(df, fname='mag_vs_re.pdf'):

    slope, intercept, r_value, p_value, std_err = stats.linregress(df["Mr"], df["Re"])

    if fname:
        label = f"$R_e = {slope:.3f} M_r + {intercept:.3f}$" if intercept>0 else \
                f"$R_e = {slope:.3f} M_r {intercept:.3f}$"
        fig   = plt.figure(figsize=(5,5))
        plt.scatter(df["Mr"], df["Re"], marker='.', s=5)
        plt.plot( [min(df["Mr"]), max(df["Mr"])],
                  [best_fit(min(df["Mr"]), slope, intercept),
                   best_fit(max(df["Mr"]), slope, intercept)],
                  c='r', label=label)
        plt.xlim(-13.75,-17.5)
        plt.ylim(  1,  6.5)
        plt.xlabel(r"$M_r \, \mathrm{(mag)}$")
        plt.ylabel(r"$R_e \, \mathrm{(kpc)}$")
        plt.legend(loc='upper left', fontsize=12)
        plt.savefig(fname, bbox_inches='tight')
        plt.close()
        
        print("\nStarndarization Formula: " + label)

    return slope, intercept

################################################################################

def stellar_mass(gr_color, M_r):
    M_sun = 4.65  # Willmer, 2018, ApJS, 236, 47
    L_r   = 10**(0.4 * (M_sun - M_r))  # Solar Luminosity

    return L_r * 10**(1.629 * gr_color - 0.792)


def axisratio_vs_re(df, select=None, bins=4, standardized=True,
                    fname='ar_vs_re.pdf', mag_vs_re_fname=None):

    # Format Scientific Notation
    def format_scinot(num, precision=2):
        exponent = int(np.floor(np.log10(num)))
        factor   = round(num/10**exponent, precision)
        return r"{0:.2f} \times 10^{1}".format(factor, exponent)

    re     = "new_Re" if standardized else "Re"
    xlabel = r"$r'_e$" if standardized else r"$r_e \, (\mathrm{kpc})$"

    # Standardized Radius + Stellar Mass
    m, b         = mag_vs_re(df, fname=mag_vs_re_fname)  # Regression Parameters
    bins         = 1 if standardized else bins           # Number of Bins
    df["new_Re"] = df["Re"] / best_fit(df["Mr"], m, b)   # Standardized Re
    df_high      = df.loc[ df["Density"] == "High" ].sort_values("Re", ascending=False)
    df_low       = df.loc[ df["Density"] == "Low"  ].sort_values("Re", ascending=False)
    

    # Create Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,8))
    size_range = np.array([min(df[re]), max(df[re])])
    ar_range   = np.array([0.3,1])
    plt.subplots_adjust(hspace=0.1)
    
    # Plot data
    ax1.scatter(df_high[re], df_high["b/a"],
                color='green', ec='k', marker='^', s=20*df_high["Re"]**2,
                lw=[1 if size<3.5 else 2.5 for size in df_high["Re"]] )
    ax2.scatter(df_low[re], df_low["b/a"],
                color='orange', ec='k', marker='o', s=20*df_low["Re"]**2,
                lw=[1 if size<3.5 else 2.5 for size in df_low["Re"]] )
    
    # Plot regression
    for ax, df_env in ((ax1, df_high), (ax2, df_low)):
        slope, intercept, r_value, p_value, std_err = stats.linregress(df_env[re], df_env["b/a"])
        expectation = best_fit(size_range, slope, intercept)
        
        if p_value < 0.05:
            ax.plot(size_range, expectation, c='green')

        print("\n\nEnvironment:  ", df_env["Density"].iloc[0])
        print("y = m x + b:   <b/a> =", slope, "r'e +", intercept)
        print("Corr Coeff :       r =", r_value)
        print("P-value:           p =", p_value)
        print("Standard Dev:      d =", std_err, "\n")
        
        print(df_env[["NAME", "Re", re, "b/a"]])

    # Set Limits
    ax1.set_ylim(ar_range)
    ax2.set_ylim(ar_range)
    ax2.set_xlim(0.6, 1.6)

    # Set Axis Labels
    ax2.set_xlabel(xlabel,   fontsize=24)
    ax1.set_ylabel(r"$b/a$", fontsize=24)
    ax2.set_ylabel(r"$b/a$", fontsize=24)
    
    # Unique Markers in Legend Only (Uses Markers w/o Bold Outline)
    legend_elements = [ Line2D( [0], [0], marker='^', color='g', mec='k', lw=1,
                                label=r"$\mathrm{High}$", markersize=np.sqrt(20*1.5**2)),
                        Line2D( [0], [0], marker='o', color='orange', mec='k', lw=0,
                                label=r"$\mathrm{Low}$", markersize=np.sqrt(20*1.5**2)) ]
    
    plt.legend(handles=legend_elements,
               title=r"$\mathrm{Environment \, Density}$",
               bbox_to_anchor=(0.7, -0.25),
               fontsize=14, fancybox=True, shadow=True, ncol=2)

    # Save Figure
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

################################################################################

if __name__ == "__main__":

    df = read_data("../data/kadowaki2019.tsv")
    df = df.loc[df["Re"]<9.0]
    
    axisratio_vs_re(df, select=None, standardized=True,
                    fname='../plots/ar_vs_re.pdf',
                    mag_vs_re_fname='../plots/mag_vs_re.pdf')
