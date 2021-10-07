import warnings
warnings.filterwarnings("ignore")

import sys
import os
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score


sys.path.insert(0, "utils/")

""" Import all python file model_config"""
import models_config
""" Import all python files in utils folder """
import utils

""" List of Directories """
Input_DIR = 'output/preprocess'
Output_DIR = 'output/prediction'
Evaluation_DIR = 'output/evaluation'

if not os.path.exists(Output_DIR):
    os.makedirs(Output_DIR)

if not os.path.exists(Evaluation_DIR):
    os.makedirs(Evaluation_DIR)


def training ():
    """
    Saves CSV file of LOOCV cross validations
    and CSV files of preditions with proper naming 
    for re-evaluation.

    """
    # dictionary for storage
    all_prediction = {}
    all_actual = {}
    all_metricCV = {}
    all_metricCV_Final = {}
    id_list = []

    for filename in os.listdir(Input_DIR):  # Goes through all the files in input directory
        if filename.endswith(".csv") : # checks the CSV files
            all_prediction[filename[:-4]], all_actual[filename[:-4]],all_metricCV[filename[:-4]], id_list = trainIndividually(filename,id_list)  # Individually train per file name: output is collected per file name
            print("---------------")
            print("---------------")
            print(filename + " predicted")

    for i in all_prediction:
        for j in all_prediction[i]:
            output_name ="[" + i + "]_["+ j + "]"  # name of CSV file
            all_metricCV_Final[output_name] = all_metricCV[i][j] 
            print(output_name)
            # results
            # combine to a temporary diction for the actual and prediction
            tempt_prediction = all_prediction[i][j].copy()
            tempt_actual = all_actual[i][j].copy()
            tempt_dict = {"Actual":tempt_actual,"ID":id_list}
            tempt_dict['Prediction'] = tempt_prediction
            df = pd.DataFrame (tempt_dict)
            utils.saveData(Output_DIR, output_name, df)
            print(output_name + " SAVED")

    df = pd.DataFrame(all_metricCV_Final).T  # save cross validation into csv file
    df = df.reset_index()
    utils.saveData(Evaluation_DIR, "LOOCV Evaluation", df)

    return 



def trainIndividually (filename, id_list ):
    """
    Splits data into train-test and then fit_predict and does cross validation
    Input:
        filename: str
            filename of the 
        id_list: list
            number of runs
    Output:
        prediction: dictionary
            dictionary of predictions
        actual: dictionary
            dictionary of actual labels (y-test)
        metricCV: dictionary
            dictionary of LOOCV results
        id_list : list
            list of patient ID 
    """
    file_description = models_config.fileDescription(filename)  # fileDescription: gets the preprocessing description from the file name
    
    print(filename)

    file_location = os.path.join(Input_DIR, filename)
    dataframe = pd.read_csv(file_location)

    prediction = {}
    actual = {}
    metricCV = {}
    
    model_description = models_config.modelDescription()

    df_copy = dataframe.copy()
    if len(id_list) == 0:
        id_list = df_copy['Patient Identification Number'].values.copy()
        print( "File ID  Copied from " + filename)
    else:
        tempt_list_1 = df_copy['Patient Identification Number'].values.copy()
        tempt_list_2 = id_list.copy()
        if len(tempt_list_1)== len(tempt_list_2) and len(tempt_list_1) == sum([1 for i, j in zip(tempt_list_1, tempt_list_2) if i == j]):
            print ("The ID lists of "+filename+ "are identical to previous")
            print("_______________________________________________________")
        else:
            print("Error !!!  Recalibrate all ID Files")

    df_copy = df_copy.drop("Patient Identification Number", axis=1)
    X, Y, features = utils.labelSeparator(df_copy,file_description['label'])  # separates labels and features from dataframe


    for i in model_description:  # goes through all the models # for Hyp 1: only regression models
        print("-------------------")
        print("-------------------")        
        print(i['model_name'], " prediction !!!")

        metricCV[i['model_name']], prediction[i['model_name']], actual[i['model_name']] = trainCV(X, Y, i['model'])  # cross validation of training
        print(i['model_name'], " prediction DONE!!!")
    
    return prediction, actual, metricCV, id_list

def trainCV(X,Y, model):
    """
    Cross validation using LOOCV
    Input:
        X: matrix
            x-train
        Y: list
            y-train
        model: model function
            model
    Output:
        metricCV: dictionary
            cross validation results
        y_pred: list
            list of prediction using the specific and model imported file
        y_true:list
            list of actual value using the specific model and imported file

    """

    metricCV = {}

    # create loocv procedure
    cv = LeaveOneOut()
    # enumerate splits
    y_true, y_pred = list(), list()
    for train_ix, test_ix in cv.split(X):
        # split data
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = Y[train_ix], Y[test_ix]
        model.fit(X_train, y_train)
        # evaluate model
        yhat = model.predict(X_test)
        # store
        y_true.append(y_test[0])
        y_pred.append(yhat[0])


    metricCV["accuracy"]= accuracy_score(y_true,y_pred)
    metricCV["sensitivity"] = recall_score(y_true,y_pred)
    metricCV["specificity"] = recall_score(y_true,y_pred, pos_label = 0 )
    metricCV["auc"] = roc_auc_score(y_true,y_pred)

    return metricCV, y_pred, y_true




if __name__ == "__main__":
  print("Starting")

  training()

  print("Done")