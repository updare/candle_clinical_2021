# Extract code here
from pathlib import Path
import pandas as pd
import numpy as np


def catFeatures (dependencies_file_path):

    """
    Creates a list of categorical features

    Input:
        dependencies_file_path: string
            string path of feature description dependency
    Output:
        catFeatures: list
            list of categorical features
    """

    dataframe = pd.read_csv(dependencies_file_path)
    tempt = dataframe.groupby(by='Feature type')
    tempt.groups.keys()
    categorical = tempt.get_group('Categorical').reset_index()
    catFeatures = categorical['Features'].tolist()
    return catFeatures


def numFeatures (dependencies_file_path):

    """
    Creates a list of numerical features

    Input:
        dependencies_file_path: string
            string path of feature description dependency
    Output:
        numFeatures: list
            list of numerical features
    """


    dataframe = pd.read_csv(dependencies_file_path)
    tempt = dataframe.groupby(by='Feature type')
    tempt.groups.keys()
    numerical = tempt.get_group('Numerical').reset_index()
    numFeatures = numerical['Features'].tolist()
    return numFeatures


def getLabels (labels_file_path):

    """
    Creates a list of possible labels 

    Input:
        labels_file_path: string
            string path of possible labels dependency
    Output:
        labels: list
            list of possible labels
    """

    label = pd.read_csv(labels_file_path, header = None )
    labels = label[0].tolist()
    return labels



