from pathlib import Path
import pandas as pd
import numpy as np
import os



def readData(input_file_path, visit_number=None):
    """
    Reads clinical (tabular) data and segregates visit X or 
    outputs the entire dataset
    
    Input:
        input_file_path: str
            file of the path to be read
        visit_number: int: default None
            if None:
                The entire dataset is outputted
            if int:
                The selected visit number is outputted
    Output:
        dataframe: pandas DataFrame
            clinical data of the selected visit dataframe
            with index number
    """
    
    dataframe = pd.read_csv(Path(input_file_path))
    if visit_number == None:
        return dataframe
    else:
        visit_check = visit_number
        return dataframe.loc[dataframe['Event Name'] == visit_check]


def combineData(demographic_file_path, clinical_data):
    """
    Combines clinical and demograophic data based on index
    and outputs combined dataframe
    
    Input:
        demographic_file_path: str
            File path of the demographic data
        clinical_data: pandas DataFrame
            Clinical data pandas DataFrame
    Output:
        clinical_data: pandas DataFrame
            Combined data with pandas DataFrame
            with index number
    """
    
    demographic_data = pd.read_csv(Path(demographic_file_path), index_col=0)
    demographic_data = demographic_data.drop('Event Name', axis=1)
    clinical_data = clinical_data.set_index('Patient Identification Number').join(demographic_data)
    return clinical_data.reset_index()


def saveData(output_file_path, output_file_name, input_dataframe):
    """
    Saves any pandas DataFrame with NO INDEX into CSV format.
    
    Input:
        output_file_path: string
            File path where file will be stored.
        output_file_name: string
            File name of the output file
        input_dataframe: Pandas DataFrame
            DataFrame to be saved.
    Output:
        None
    
    """
    
    file_path_and_name = os.path.join(output_file_path, output_file_name)
    input_dataframe.to_csv("{}.csv".format(file_path_and_name), index=False)
    print("{}.csv File is Saved!".format(output_file_name))
    print("----------------------------------")
    return


def labelSeparator (dataset, label):

    dataframe = dataset.copy()

    features = len(dataset.columns) - 1 

    Y = dataframe[label].values


    Y = Y.astype('int')

    X = dataframe.drop([label], axis = 1).values

    return X, Y, features 
