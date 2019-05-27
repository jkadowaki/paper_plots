#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':12,'text.usetex':True})


################################################################################

# EM0 for BERT+All Data: Table 1 #10
bert_all = 17.00
lab_all  = r"$\mathrm{BERT+entire \, passage}$"

# EM0 for BERT+BM25 Data: Table 1 #11-15
xlim1    = range(1,6)
bm25     = np.array([17.94, 20.99, 21.62, 22.35, 23.40])
lab_bm25 = r"$\mathrm{BERT+BM25}$"

# EM0 for BERT+ROCC Data: Table 1 #16-19
xlim2    = range(2,6)
rocc     = np.array([22.67, 25.18, 24.97, 22.67])
lab_rocc = r"$\mathrm{BERT+ROCC}$"

# EM0 for BERT+Auto-ROCC(2,3,4,5,6): Table 1 #22
bert_auto = 25.29
lab_auto  = r"$\mathrm{BERT+AutoROCC}$"

# F1 for BERT+AutoROCC Data: Table 2 #21
xlim = range(2,21)
bert_auto_f1 = 54.32
lab_auto_f1  = r"$\mathrm{BERT+AutoROCC}$"

# F1 for BERT+BM25 Data: Table 2 #22
bert_bm25_f1 = 48.31
lab_bm25_f1  = r"$\mathrm{BERT+BM25}$"

# F1 for BERT+AutoROCC, pre-trained Data: Table 2 #23
bert_pre_f1  = 54.32
lab_pre_f1   = r"$\mathrm{BERT+AutoROCC, \, pre}$-$\mathrm{trained}$"


################################################################################

def plot1(fname="vikas_plot1.pdf"):

    """
    PLOT1: Creates a figure with 2 side-by-side plots.
    
    Args:
        fname (str): Name of figure.
    
    Returns:
        None.
    """


    fig, (ax1,ax2) = plt.subplots(1,2,sharey=False, figsize=(10,5))
    
    ############################## SUBPLOT 1: EM0 ##############################
    
    ax1.plot(xlim1, bm25, label=lab_bm25, marker=".", linewidth=2)
    ax1.plot(xlim1, [bert_all]*len(xlim1), label=lab_all, marker="d", linewidth=2)
    ax1.plot(xlim2, rocc, label=lab_rocc, marker="^", linewidth=2)
    ax1.plot(xlim2, [bert_auto]*len(xlim2), label=lab_auto, marker="x", linewidth=2)

    xrange = min(xlim1[0], xlim2[0]), max(xlim1[-1], xlim2[-1])
    xticks = range(xrange[0], xrange[1]+1)
    ax1.set_xlim(xrange)
    ax1.set_xticks(xticks, [r'$'+ str(x) + r'$'  for x in xticks])
    
    ax1.set_xlabel(r"$\mathrm{\# \, of \, Sentences}$")
    ax1.set_ylabel(r"$\mathrm{EM0}$")
    ax1.set_title(r"$\mathrm{Performance}$")
    ax1.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.4), fontsize=10)
    
    
    ############################## SUBPLOT 2: F1 ###############################
    
    ax2.plot(xlim, [bert_bm25_f1]*len(xlim), label=lab_bm25_f1, marker=".")
    ax2.plot(xlim, [bert_auto_f1]*len(xlim), label=lab_auto_f1, marker="x")
    ax2.plot(xlim, [bert_pre_f1]*len(xlim),  label=lab_pre_f1,  marker="^")
    
    ax2.set_xlabel(r"$\mathrm{\# \, of \, Sentences}$")
    ax2.set_ylabel(r"$\mathrm{F1}$")
    ax2.set_title(r"$\mathrm{Performance}$")
    ax2.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.4), fontsize=10)
    
    
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

################################################################################

def plot2(fname="vikas_plot2.pdf"):
    pass

################################################################################

def main():
    plot1()

################################################################################

if __name__ == '__main__':
    main()
