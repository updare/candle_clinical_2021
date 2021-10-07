import warnings

warnings.filterwarnings("ignore")

import os
import sys

import pandas as pd

sys.path.insert(0, "data_preprocessing/extract/")
sys.path.insert(0, "data_preprocessing/pre_processing/")
sys.path.insert(0, "utils/")

""" Import all python files in extract folder """
import extract

""" Import all python files in pre_processing folder """
import data_preprocessing
import remove
import standardize

""" Import all python files in utils folder """
import utils

""" List of Directories """
Input_DIR = "input"
Output_DIR = 'output/preprocess'
Dependency_DIR = "dependencies"

if not os.path.exists(Output_DIR):
    os.makedirs(Output_DIR)


def prepropAll(lab_path):
    """ 
    Preprocess everything
    Input:
        lab_path: str
            directory of file names
    Output:
        CSV files for each preprocessing combination with proper naming

    """
    ## Opens the necessary files from dependencies and input
    lab_df = pd.read_csv(lab_path)

    cat_Features = extract.catFeatures (os.path.join(
                Dependency_DIR, 
                "feature_description.csv"))  # open categorical features

    num_Features = extract.numFeatures (os.path.join(
                Dependency_DIR, 
                "feature_description.csv"))  # open numerical features

    label_list = ['Baseline']
    
    cat_Features = [item for item in cat_Features if item not in label_list]

    numerical_preprop = ['null','normalize','standardize']

    categorical_preprop = ['null','onehot']

    evaluation_preprop = ['null']

    visit_list = ['Visit 1']

    # iteration proper
    for visit in visit_list:

        df = lab_df.copy()

        df = standardize.basicStandardization(df)

        if visit != 'Visit 1':
            tempt = [] # temporary storage of labels_list: Appends all the baseline labels to one list
            for i in label_list:
                if i.endswith('Baseline'):
                    tempt.append(i)
            label_list = tempt.copy()

        for num in numerical_preprop:
            for cat in categorical_preprop:
                # data preprocessing of numerical and categorical features
                df_2 = numerical_preprocess(df, num, num_Features)
                df_2 = categorical_preprocess(df_2,cat, cat_Features)
                # feature selection
                # For hypothesis 1: no feature selection is done
                for eva in evaluation_preprop:
                    # label selection
                    # For hypothesis 1: labels are only baseline: Normal v Abnormal
                    for labe in label_list:
                        if labe in df_2.columns:
                            if df_2[labe].nunique() > 1:
                                if (eva == 'chi' and num == 'normalize') is False:
                                    df_3 = evaluation_preprocess (df_2,eva,labe)
                                    output_filename = visit + "_"+ num + "_"+ cat + "_"+ eva + "_"+ labe
                                    
                                    try:
                                        if (len(df_3.columns) - 1) > 0:
                                            utils.saveData(Output_DIR, output_filename, df_3)
                                    except AttributeError:
                                        print("{}.csv File is NONETYPE!".format(output_filename))
                                        print("----------------------------------")

    return


def numerical_preprocess (dataframe, type, num_Features):
    """
    Applies numerical preprocessing based on input
    Input:
        dataframe: pandas DataFrame
            dataframe to be processed
        type: str
            different types of preprocessing
        num_features: list
            list of numerical features
    Output:
        df: pandas DataFrame
            preprocessed file
            index: None
    """

    df = dataframe.copy()

    if type == 'null':
        return df
    elif type == 'normalize':
        df = data_preprocessing.normalizeNumerical(df,num_Features)
        return df
    elif type == 'standardize':
        df = data_preprocessing.standardizeNumerical(df,num_Features)
        return df


def categorical_preprocess (dataframe,type, cat_Features):
    """
    Applies categorical preprocessing based on input
    Input:
        dataframe: pandas DataFrame
            dataframe to be processed
        type: str
            different types of preprocessing
        cat_features: list
            list of categorical features
    Output:
        df: pandas DataFrame
            preprocessed file
            index: None
    """

    df = dataframe.copy()

    if type == 'null':
        return df
    elif type == 'onehot':
        df = data_preprocessing.oneHotEncoding(df,cat_Features)
        return df


def evaluation_preprocess (dataframe,type,label):
    """
    Applies feature selection based on input
    Input:
        dataframe: pandas DataFrame
            dataframe to be processed
        type: str
            different types of feature selection
        label: str
            label
    Output:
        df: pandas DataFrame
            preprocessed file
            index: None
    """

    df = dataframe.copy()

    if type == 'correlation':
        df = data_preprocessing.corrAttrEval(df, label, ratio_included=0.5)
        return df
    elif type == 'information':
        df = data_preprocessing.infoGainAttrEval(df, label, ratio_included=0.5)
        return df
    elif type == 'adaboost':
        df = data_preprocessing.adaboostFeatureSelect(df, label, ratio_included=0.5)
        return df
    elif type == 'random':
        df = data_preprocessing.randomForestPrep(df, label)
        return df
    elif type == 'chi':
        df = data_preprocessing.chi2PreProp(df, label, ratio_included=0.0055)
        return df
    elif type == 'null':
        df = df.copy()
        return df


## MAIN
if __name__ == "__main__":
  print("starting")
  filename = os.path.join(
    Input_DIR, 
    "Data.csv")

  prepropAll(filename)