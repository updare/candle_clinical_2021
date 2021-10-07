import warnings
warnings.filterwarnings("ignore")

import sys
import os
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split

sys.path.insert(0, "../../utils/")

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score


#""" Import all python file model_config"""
#import models_config
""" Import all python files in utils folder """
import utils

""" List of Directories """
Input_DIR = 'output/prediction'
Output_DIR = 'output/evaluation'

if not os.path.exists(Output_DIR):
    os.makedirs(Output_DIR)

def generateScore (prediction, actual):
    """
    Splits data into train-test and then fit_predict and does cross validation
    Input:
        prediction: list
            list of prediction using the specific and model imported file
        actual:list
            list of actual value using the specific model and imported file
    Output:
        metric: dictionary
            dictionary containing the summary of the performance given the specific 
    """
    metric = {}

    metric["accuracy"]= accuracy_score(actual,prediction)
    metric["sensitivity"] = recall_score(actual,prediction)
    metric["specificity"] = recall_score(actual,prediction, pos_label = 0 )
    metric["auc"] = roc_auc_score(actual,prediction)

    return metric

def evaluate():
    """
    Runs the previously saved predictions through different performance 
    metrics for re-evaluation
    Input:
    Output:
    """


    all_evaluation = {}
    for filename in os.listdir(Input_DIR):
        if filename.endswith(".csv") : 
            all_evaluation[filename[:-4]] = evaluateIndividually(filename)


    df = pd.DataFrame(all_evaluation).T
    df = df.reset_index()
    utils.saveData(Output_DIR, "Evaluation Result", df)

    return


def evaluateIndividually(filename):
    """
    runs the previously saved predictions through different performance 
    metrics for re-evaluation per filename
    Input:
        filename: str
            string filename of the  previously saved predictions
    Output:
        metric: dictionary
            dictionary containing the summary of the performance given the specific 

    """

    file_location = os.path.join(Input_DIR, filename)
    dataframe = pd.read_csv(file_location)
    metric = generateScore(dataframe["Prediction"].values,dataframe["Actual"].values)
    return metric 



if __name__ == "__main__":
  print("Starting")
  evaluate()
  print("Done") 