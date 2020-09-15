#!/usr/bin/env python

################################################################################

import sys
sys.path.append("../code")

from read_data import read_data
import matplotlib.cm as cm
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
        
        print("\nStarndarization Formula:" + label)

    return slope, intercept

################################################################################

def stellar_mass(gr_color, M_r):
    M_sun = 4.65  # Willmer, 2018, ApJS, 236, 47
    L_r   = 10**(0.4 * (M_sun - M_r))  # Solar Luminosity

    return L_r * 10**(1.629 * gr_color - 0.792)


def axisratio_vs_re(df, select=None, bins=5, standardized=True,
                    fname='ar_vs_re.pdf', mag_vs_re_fname=None):

    # Format Scientific Notation
    def format_scinot(num, precision=2):
        exponent = int(np.floor(np.log10(num)))
        factor   = round(num/10**exponent, precision)
        return r"{0:.2f} \times 10^{1}".format(factor, exponent)

    re     = "new_Re" if standardized else "Re"
    xlabel = r"$\mathrm{Standardized} \, R_e$" if standardized else r"$R_e \, (\mathrm{kpc})$"

    # Standardized Radius + Stellar Mass
    m, b         = mag_vs_re(df, fname=mag_vs_re_fname)  # Regression Parameters
    bins         = 1 if standardized else bins           # Number of Bins
    df["new_Re"] = df["Re"] / best_fit(df["Mr"], m, b)   # Standardized Re
    df["M_star"] = stellar_mass(df["g-r"], df["Mr"])     # Stellar Mass
    df["color"]  = ['g' if obj=="High" else 'orange' for obj in df["Density"]]
    df["marker"] = ['^' if obj=="High" else 'o'      for obj in df["Density"]]
    
    # Compute Stellar Mass Bounds
    min_mass = int(np.floor(np.log10(min(df["M_star"]))))  # Lower Limit
    max_mass = int(np.ceil(np.log10(max(df["M_star"]))))   # Upper Limit
    bin_edge = np.logspace(min_mass, max_mass, bins+1)     # Bounds
    colors   = cm.rainbow(np.linspace(0, 1, bins))         # Colors
    print("Bins:", bins, bin_edge)

    # Create Figure
    fig, ax    = plt.subplots(figsize=(8,8))
    size_range = np.array([min(df[re]), max(df[re])])
    ar_range   = np.array([0.3,1])

    # Iterate through all bins & plot with respective format
    for idx in range(bins):
        # Select appropriate data frame
        print(f"\n\nBin {idx}: {bin_edge[idx]:.2e} to {bin_edge[idx+1]:.2e}")
 
        #try:
        df_select = df.loc[ (df["M_star"] < bin_edge[idx+1]) & (df["M_star"] >= bin_edge[idx])]
        print(df_select)
        # Plot individual data
        for index, udg in df_select .iterrows():
            label = r"$\mathrm{{{0}}}$".format(udg["Density"]) if standardized else \
                    r"${0} \leq M_* < {1}$".format( format_scinot(bin_edge[idx], 2),
                                            format_scinot(bin_edge[idx+1],2) )
            ax.scatter( udg[re], udg["b/a"],
                        color=udg["color"] if standardized else colors[idx],
                        marker=udg["marker"],
                        label=label)
            
        # Plot regression
        for df_env in (df.loc[df["Density"]=="High"], df.loc[df["Density"]=="Low"]):
            slope, intercept, r_value, p_value, std_err = stats.linregress(df_env[re], df_env["b/a"])
            expectation = best_fit(size_range, slope, intercept)
            ax.plot(size_range, expectation, c=df_env.iloc[0]["color"] if standardized else colors[idx])


        #except:
        #    print("No objects in this bin.")

    # Set Limits
    ax.set_ylim(ar_range)

    # Set Axis Labels
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(r"$b/a$", fontsize=24)
    
    # Unique Markers in Legend Only (Uses Markers w/o Bold Outline)
    handles, labels = ax.get_legend_handles_labels()
    unique_index    = [labels.index(l) for l in set(labels)]
    unique          = [(h,l) for i, (h,l) in enumerate(zip(handles, labels))
                            if  i in unique_index]
    if standardized:
        ax.legend(*zip(*unique),
                      title=r"$\mathrm{Environment \, Density}$",
                      loc='lower right',
                      fontsize=14, fancybox=True, shadow=True)
    else:
        ax.legend(*zip(*unique),
                  title=r"$\mathrm{Stellar \, Mass}$",
                  bbox_to_anchor=(0.5, -0.125),
                  fontsize=14, fancybox=True, shadow=True, ncol=2)

    # Save Figure
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

################################################################################

if __name__ == "__main__":

    df = read_data("../data/kadowaki2019.tsv")
    df = df.loc[df["Re"]<9.0]
    
    axisratio_vs_re(df, select=None, bins=4, standardized=True,
                    fname='../plots/ar_vs_re.pdf',
                    mag_vs_re_fname='../plots/mag_vs_re.pdf')
