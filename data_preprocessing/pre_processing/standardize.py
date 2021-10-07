# standardize code here
from pathlib import Path
import pandas as pd
import numpy as np


def encodeLabels (dataframe, labels_list):

    """
    Encodes string values into numeric values

    Input:
        dataframe : Pandas Dataframe
            combined dataframe demographic and clinical data
        labels_list: list
            a list of all the possible labels 
    Output:
        dataframe : Pandas Dataframe
            dataframe with encoded labels 
    """

    """Separates the categorical features of demographic and clinical data"""
    interpretation = []
    baseline = []
    for i in labels_list:
        if i.endswith('interpretation'):
            interpretation.append(i)
        elif i.endswith('Baseline'):
            baseline.append(i)

    """All categorical features of clinical data have the same format"""
    for i in interpretation:
        if i in dataframe.columns:
            dataframe.loc[(dataframe[i] == 'REACTIVE'), i] = 1
            dataframe.loc[(dataframe[i] == 'NONREACTIVE'), i] = 0
            dataframe.loc[(dataframe[i] != 1 ) & (dataframe[i] != 0 ), i] = 2


    """Creates a new column that combines the baseline info of Hepatitis B
    and then drops the two excess features"""
    
    dataframe['Baseline'] = 0
    dataframe.loc[(dataframe['HBsAG Baseline'] == 'HBsAg positive') |
            (dataframe['Cirrhosis Baseline'] == 'Yes - with cirrhosis') |
            (dataframe['HCC Baseline'] == 'Yes - with HCC'), 'Baseline'] = 1

    extra = ['HBsAG Baseline','Anti-HBc Baseline','Cirrhosis Baseline','HCC Baseline']
    for i in extra: 
        dataframe = dataframe.drop(i, axis=1)

    dataframe.loc[(dataframe['Gender'] == 'Female'),'Gender'] = 0
    dataframe.loc[(dataframe['Gender'] == 'Male'),'Gender'] = 1


    return dataframe


def basicStandardization (dataframe):

    """
    Changes undetectable and out of bound values to numerical numbers

    Input:
        dataframe : Pandas Dataframe
            data before preprocessing
    Output:
        dataframe : Pandas Dataframe
            dataframe with converted standardized values

    """


    df_updated = dataframe.copy()
    if 'HBV Viral Load (IU/mL) ' in dataframe.columns:
        df_updated['HBV Viral Load (IU/mL) '] = df_updated['HBV Viral Load (IU/mL) '].replace(to_replace ='^<[ ]?20', value = 19, regex = True)
        df_updated['HBV Viral Load (IU/mL) '] = df_updated['HBV Viral Load (IU/mL) '].replace(to_replace ='^>[ ]?170,000,000', value = 170000001, regex = True)
        df_updated['HBV Viral Load (IU/mL) '] = df_updated['HBV Viral Load (IU/mL) '].replace(to_replace ='Not tested', value = 0, regex = False)
        df_updated['HBV Viral Load (IU/mL) '] = df_updated['HBV Viral Load (IU/mL) '].replace(to_replace ='Not detected', value = 0, regex = False)
    

    df_updated = df_updated.replace(to_replace ="***", value = 0, regex = False)
    df_updated = df_updated.replace(to_replace ="^<[ ]?(\d+(\.\d+)?)", value = 0, regex = True) 
    df_updated = df_updated.replace(to_replace ="^>[ ]?(\d+(\.\d+)?)", value = 99999, regex = True) 

    return df_updated

