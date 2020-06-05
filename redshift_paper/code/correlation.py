#!/usr/bin/env python

from __future__ import print_function
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 16}
plt.rc('font', **font)

import argparse
import numpy as np
import os
import pandas as pd
from scipy import stats
from sklearn import linear_model
import seaborn as sns
import sys

sys.path.append('../code')
from read_data import read_data


################################################################################

def parse_args(verbose=True):
    """
    Parses Command Line Arguments
    """
    ######################   Checks for Valid Arguments   ######################
    
    def is_valid_file(parser, arg):
        if not os.path.exists(arg):
            parser.error("The file {0} does not exist!".format(arg))
        else:
            return str(arg)

    ############################################################################

    parser = argparse.ArgumentParser(
                 description="Generates Corelation & p-value Matrix")
    
    # Required Arguments
    parser.add_argument('data', help="Data File", metavar="FILE",
                        type=lambda x: is_valid_file(parser, x))
    
    # Optional Arguments
    parser.add_argument('--table', nargs='+',
                        default=['all'],
                        choices=['2','3','4','smudges','all'],
                        help="Choose one or more from {2,3,4,smudges,all} " \
                             "separated with spaces.")

    parser.add_argument('--environment', nargs='+',
                        default=['all'],
                        choices=['local','global','density','all'],
                        help="Choose one or more from {local,global,density} " \
                             "separated with spaces.")

    # Flag to select UDGs/candidates
    parser.add_argument('--udgs',       dest='udgs_only', action='store_true')
    parser.add_argument('--candidates', dest='udgs_only', action='store_false')
    
    # Flag to run script verbosely/silently
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--silent',  dest='verbose', action='store_false')
    
    # Set Defaults & Return Command Line Arguments
    parser.set_defaults(udgs_only=True, verbose=False)
    args = parser.parse_args()

    if verbose:
        print("\nParameters:", args)

    return args


################################################################################
#
#def fisher_transform(r):
#    """
#    Performs the Fisher's z-tranformation of rank correlation r.
#    Sample distribution of r is expected to be normal with a stable variable,
#    which indicates the Fisher transformation can be used to test hypothesis on
#    the population correlation coefficient.
#    See More: https://en.wikipedia.org/wiki/Fisher_transformation
#    Implementation: Numerical Recipes in C (14.5 Linear Correlation, Page 639)
#    """
#    return 0.5 * np.log((1.0+r+TINY) / (1.0-r+TINY))
#
################################################################################
#
#def confidence_interval(r):
#    """
#    Computes confidence interval from Fisher Transform.
#    r (float): Estimate of the strength of correlation
#    """
#    def limit(r, bound='lower'):
#        """
#        Computes the 'lower' or 'upper' confidence interval bounds.
#        bound (str): 'lower' or 'upper' bound]
#        """
#        sign = -1 if   bound == 'lower' \
#                  else 1 if bound == 'upper' \
#                         else 0
#        return np.tanh(np.arctanh(r) + sign * 1.06 / np.sqrt(n-3))
#    return limit(r, bound='lower'), limit(r, bound='upper')
#
################################################################################
#
# TODO:
# 1. Bootstrap
# 2. Compute the Fisher Transformation
# 3. Get the transformed 95% confidence intervals
# 4. Inverse Fisher Transform the confidence limits.
#
################################################################################

def test_statistic(coeff, num):
    """
    Computes the test statistic for the t-distribution with n-2 degrees of freedom
    """
    with np.errstate(divide='ignore'):
        return coeff * np.sqrt(num-2) / np.sqrt(1 - coeff**2)


################################################################################

def get_data(data_file, table=[2,3,4], udgs_only=True, field='Coma', environment='local', verbose=True):

    """
    Args:
        data_file
        table (list of int):
        udgs_only
        field
        environment
    Returns:
        (dictionary)
    
    """
    if verbose:
        print('\n{0}\n'.format('-'*150))
        print("File:       ", data_file)
        print("Table:      ", table)
        print("Objects:    ", "UDGs" if udgs_only else "Candidates")
        print("Environment:", environment, '\n')

    # Load Data
    df_results = read_data(data_file, udg_only=udgs_only, field=field)
    df_subset  = df_results.loc[df_results["TABLE"].isin(table)]
   
    df_dict = { 'sparse':      None,   'dense':   None,    # Local Environment
                'non-cluster': None,   'cluster': None,    # Global Environment
                'low':         None,   'high':    None,    # Environment Density
                'all':         None }

    for env in environment:
        if env.lower() == 'all':
            df_subset = df_subset.reset_index(drop=True)
            df_dict['all'] = df_subset

        if env.lower() == 'local':
            df_subset = df_subset.sort_values(by=['LocalEnv'])
            df_subset = df_subset.reset_index(drop=True)
            df_dict['sparse'] = df_subset.loc[df_subset['LocalEnv'] == "Sparse"]
            df_dict['dense']  = df_subset.loc[df_subset['LocalEnv'] == "Dense"]

        if env.lower() == 'global':
            df_subset = df_subset.sort_values(by=['GlobalEnv'])
            df_subset = df_subset.reset_index(drop=True)
            df_dict['non-cluster'] = df_subset.loc[df_subset['GlobalEnv'] == "Non-Cluster"]
            df_dict['cluster']     = df_subset.loc[df_subset['GlobalEnv'] == "Cluster"]

        if env.lower() == 'density':
            df_subset = df_subset.sort_values(by=['Density'])
            df_subset = df_subset.reset_index(drop=True)
            df_dict['high'] = df_subset.loc[df_subset['Density'] == "High"]
            df_dict['low']  = df_subset.loc[df_subset['Density'] == "Low"]

    return df_dict


################################################################################

def compute_correlation(df_dict, feature1, feature2, verbose=True):
    """
    
    """

    if verbose:
        print("Testing Correlations between:", feature1.rjust(6), "and", feature2)

    correlation = []

    for key, df in df_dict.items():
        if df is not None:
            df_relevant  = df[[feature1, feature2]].dropna()
            rho, p_value = stats.spearmanr(df_relevant)
            t_stat       = test_statistic(rho, len(df_relevant))

            correlation.append( {'environment': key, 
                                 'feature1':    feature1,
                                 'feature2':    feature2,
                                 'num':         len(df_relevant),
                                 'rho':         rho,
                                 'p-value':     p_value,
                                 't-stat':      t_stat} )
    return correlation


################################################################################

def print_dictlist(dictlist):
    """
    Pretty prints a list of dictionaries in easily read columns.
    """
    if dictlist:
        # Get Keys & Lengths of Values
        max_length = {key:len(str(val)) for key,val in dictlist[0].items()}

        # Iterate through list and retrieve maximum value length for each key
        for dictionary in dictlist:
            for key,value in dictionary.items():
                if max_length[key] < len(str(value)):
                    max_length[key] = len(str(value))
    
        # Print Standardized Values 
        for dictionary in dictlist:
            print(  { key: str(value).ljust(max_length[key])
                      if type(value) != type(0.0)
                      else str("%.5f"%value)
                      for key,value in dictionary.items() }  )


################################################################################

def bonferroni_correction( comparisons, normalization_factor=1,
                           significance=0.05, verbose=True ):
    """
    Applies Bonferroni correction to individual hypothesis tests.
    https://en.wikipedia.org/wiki/Bonferroni_correction

    comparisons (list of dicts): List of hypothesis testing results
    significance (float): Significance level for each individual hypothesis testing
    """
     
    n = len(comparisons)
    new_significance  = significance / n / normalization_factor
    rejected_hypotheses = [ df for df in comparisons 
                            if df['p-value'] < new_significance ]
    if verbose:
        print('')
        print('Number of Tables:        ', normalization_factor)
        print('Number of Comparisons:   ', n, 'per Table')
        print('                         ', n * normalization_factor, 'Total\n')
        print('Uncorrected Significance:', significance)
        print('New Significance Level:  ', new_significance)
        print('\nRejected Hypotheses:')
        print_dictlist(rejected_hypotheses)

    return rejected_hypotheses



################################################################################

def main(data_file='kadowaki2019.tsv', table=[2,3,4],
         normalization_factor=1,
         udgs_only=True, environment='local,global,density', verbose=True):
    
    """
    Args:
        data_file  (str):
        table (int/list):
        udgs_only (bool):
        enviroment (str):
    """

    # Retrieve relevant dataframe requested by command line arguments
    df_dict = get_data(data_file=data_file, table=table, udgs_only=udgs_only,
                  environment=environment, verbose=verbose)
    
    # Stores Hypothesis Testing Comparisons
    correlation = []


    # Visits Every Combination of Features in Unique/Non-duplicated Feature Dictionary Pairs
    feature_groups = [ 'magnitudes', 'extrinsic', 'intrinsic', 'colors' ]
    feature_dict   = { 'magnitudes': ["Mnuv", "Mg", "Mr", "Mz"],
                       'colors':     ["NUV-g", "NUV-r", "NUV-z", "g-r", "g-z", "r-z"],
                       'intrinsic':  ["n", "Re", "MUg0", "b/a"],
                       'extrinsic':  ["cz", "sepMpc", "NUM500"] }
    """
    for fg_idx1, fgroup1 in enumerate(feature_groups):
        for fg_idx2, fgroup2 in enumerate(feature_groups):
            
            # Skip; Already visited.
            if fg_idx1 >= fg_idx2:
                continue
    
            # Iterate through
            for f1 in feature_dict[fgroup1]:
                for f2 in feature_dict[fgroup2]:
                    correlation.extend(
                        compute_correlation(df_dict, f1, f2, verbose=verbose) )
    """

    # User-specified pairs of features
    features      = [ 'sepMpc', 'MUg0', 'Mg', 'g-r', 'Re', 'b/a', 'n' ]
    exclude_pairs = [] #[ ('sepMpc', 'cz'), ('cz', 'sepMpc'), 
                       #  ('Re',     'Mg'), ('Mg', 'Re'),
                       #  ('g-r',    'Mg'), ('Mg', 'g-r') ]
    feature_pairs = [ (f1, f2) for idx1,f1 in enumerate(features) 
                               for idx2,f2 in enumerate(features)
                               if  idx1>idx2
                               if  (f1, f2) not in exclude_pairs ]
    
    for f1, f2 in feature_pairs:
        correlation.extend( compute_correlation(df_dict, f1, f2, verbose=verbose) )

    bonferroni_correction( correlation, normalization_factor,
                           significance=0.05, verbose=verbose )


################################################################################

if __name__ == "__main__":
    
    # EXAMPLE CALL
    """
    $CODE/correlation.py  $DATA/kadowaki2019.csv               \
                          --udgs                               \
                          --table 2 3 4 smudges all            \
                          --environment local global density   \
                          --verbose
    """
    # Parse Command Line Arguments
    args = parse_args()

    # Computes a separate correlation/p-value matrix for each data subset.
    for table in args.table:
        try:
            subset = [int(table)]
            if int(table) not in [2,3,4]:
                raise Exception("{0} is not a valid table number".format(subset))
        except:
            if table.lower() == 'smudges':
                subset = [2,3]
            elif table.lower() == 'all':
                subset = [2,3,4]
            else:
                raise Exception("{0} is not a valid table name.".format(table))

        main( data_file=args.data,
              table=subset,
              normalization_factor=len(args.table),
              udgs_only=args.udgs_only,
              environment=args.environment,
              verbose=args.verbose )

