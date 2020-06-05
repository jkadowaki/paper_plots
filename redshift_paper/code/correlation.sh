#! /bin/bash

PROJECT_DIR='/Users/jkadowaki/Documents/github/paper_plots/redshift_paper'
CODE=$PROJECT_DIR/'code'
DATA=$PROJECT_DIR/'data'
RESULTS=$PROJECT_DIR/'results'

# Run Hypothesis Testing for Correlation

#$CODE/correlation.py $DATA/kadowaki2019.tsv                 \
#                     --udgs                                 \
#                     --table 2 3 4 smudges all              \
#                     --environment local global density all \
#                     --verbose                             #\
#> $RESULTS/all_comparisons.txt


$CODE/correlation.py $DATA/kadowaki2019.tsv \
                     --udgs                 \
                     --table all            \
                     --environment all      \
                     --verbose              #\
#> $RESULTS/min_comparisons.txt


$CODE/correlation.py $DATA/kadowaki2019.tsv \
                     --udgs                 \
                     --table all            \
                     --environment local    \
                     --verbose              \
> $RESULTS/local_comparisons.txt


$CODE/correlation.py $DATA/kadowaki2019.tsv \
                     --udgs                 \
                     --table all            \
                     --environment global   \
                     --verbose              \
> $RESULTS/global_comparisons.txt


$CODE/correlation.py $DATA/kadowaki2019.tsv \
                     --udgs                 \
                     --table all            \
                     --environment density  \
                     --verbose              \
> $RESULTS/density_comparisons.txt
