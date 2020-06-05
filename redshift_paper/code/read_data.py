#!/usr/bin/env python

import pandas as pd

################################################################################

def read_data(file, udg_only=True, field=None):
    """
    Loads relevant data from file into a Pandas DataFrame.
        
    ARGS:
        file (str): Name of .tsv file containing UDG properties.
    RETURNS:
        (DataFrame):
    """
    
    dict_list = []
    columns   = []
    
    with open(file) as f:
        for idx, line in enumerate(f):
            if idx == 0:
                columns = line.split()
            else:
                dict_list.append( {key:val for key,val
                                 in zip(columns, line.split())} )

    # Converts NaNs to -1 for integer data-types
    int_features = ['NUM500', 'NUM1000', 'cz']
    for obj in dict_list:
        for feat in int_features:
            if obj.get(feat)=='NaN':
                obj[feat] = -1

    # Creates DataFrame from list of dictionaries.
    # Specifies the data type for each columns.
    df = pd.DataFrame.from_dict(dict_list).astype(
         {   'NAME':str,    'FIELD':str,      'TABLE':int,
               'ra':float,    'dec':float,       'cz':int,   'redshift':float,
           'sepMpc':float, 'sepDEG':float,   'NUM500':int,
               'Re':float,   'MUg0':float,      'b/a':float,    'n':float,
             'Mnuv':float,     'Mg':float,       'Mr':float,   'Mz':float,
            'NUV-g':float,  'NUV-r':float,    'NUV-z':float,   'UV':str,
              'g-r':float,    'g-z':float,      'r-z':float,
              'udg':str, 'LocalEnv':str,  'GlobalEnv':str,   'Density':str   } )

    # Filters out objects not in specified field
    if field:
        df = df.loc[df['FIELD']==field]

    # Return DataFrame with UDGs only if requested.
    if udg_only:
        return df.loc[df['udg']=='TRUE']
    
    return df

################################################################################
