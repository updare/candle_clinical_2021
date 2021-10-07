# Remove features here
from pathlib import Path
import pandas as pd
import numpy as np
import os



def rmFeatures (dataframe,directory,visit):
    """
    Creates a list of columns to be removed and drops them from the dataframe

    Input:
        dataframe: Pandas DataFrame
            Raw DataFrame
    Output:
        dataframe: Pandas DataFrame
            Filtered DataFrame
    """

    """List of Dependencies"""
    discontinued = pd.read_csv(os.path.join(directory, 'discontinued.csv'), header = None )
    irrelevant = pd.read_csv(os.path.join(directory,'irrelevant_features.csv'), header = None )
    many = pd.read_csv(os.path.join(directory,'many_undetectable.csv'), header = None )
    alt = pd.read_csv(os.path.join(directory,'redundant_alt.csv'), header = None )
    ast = pd.read_csv(os.path.join(directory,'redundant_ast.csv'), header = None )
    patient = pd.read_csv(os.path.join(directory,'remove_patients.csv'))
    once = pd.read_csv(os.path.join(directory,'visit_1_only.csv'), header = None) 

    """ Removes Columns based on the dependencies"""
    if visit == 'Visit 1':
        remove_column = [discontinued, irrelevant, many, alt, ast]
    else:
        remove_column = [discontinued, irrelevant, many, alt, ast, once]

    to_remove = []
    for i in remove_column:
        for j in i[0]:
            to_remove.append(j)

    """ Removes Patients based on the dependencies"""
    if visit == 'Visit 1':
        for i in patient['no_visit_1']:
            dataframe = dataframe[dataframe['Patient Identification Number'] != i]

    elif visit == 'Visit 2':
        for i in patient['no_visit_2']:
            dataframe = dataframe[dataframe['Patient Identification Number'] != i]

    dataframe = dataframe.reset_index()
    dataframe = dataframe.drop('index', axis=1)
    dataframe = dataframe.drop('Event Name', axis=1)


    for i in to_remove:
        if i in dataframe.columns:
            dataframe = dataframe.drop(i, axis=1)

    return dataframe
