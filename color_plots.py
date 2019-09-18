#!/usr/bin/env python

from __future__ import print_function
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import csv
import errno
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
import numpy as np
import os
import pandas as pd
from scipy import stats
import seaborn as sns
sns.set(font_scale=25, rc={'text.usetex':True}, style="darkgrid", color_codes=True)
sns.set_style("darkgrid", {"legend.frameon":True})

import paper2_plots


