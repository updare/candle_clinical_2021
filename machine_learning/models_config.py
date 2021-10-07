from pathlib import Path
import pandas as pd
import numpy as np
import os

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier






def fileDescription (filename):
    """
    Acquire file features based on the filename format 
    Input: 
        filename:  str
            string of the filename
    Output:
        file_features: dictionary
            dictionary of the file features

    """

    filename = filename[:-4]
    filename = filename.split("_")
    file_features = {}
    file_features['visit'] = filename[0]
    file_features['numerical'] = filename[1]
    file_features['categorical'] = filename[2]
    file_features['evaluation'] = filename[3]
    file_features['label'] = filename[4]

    return file_features


def modelDescription ():
    """
    Generate a dictionary containing model name and model parameterss

    Input: 

    Output:
        model_description: dictionary
            dictionary of different models and their respective parameters

    """

    model_description = [

        {
        "model_name" : "Logistic Regression",
        "model" : LogisticRegression(penalty = "none" , solver="lbfgs", multi_class="ovr", class_weight='balanced')
        },
        
        {
        "model_name" : "Gradient Boosting",
        "model" : GradientBoostingClassifier()
        },

        {
        "model_name" : "Random Forest",
        "model" : RandomForestClassifier()
        },

        {
        "model_name" : "Decision Tree",
        "model" : DecisionTreeClassifier()
        }
    ]
    return model_description


if __name__ == "__main__":
  filename = 'Visit 1_normalize_null_random_Anti-HBc total interpretation.csv'

  dictionary = fileDescription (filename)
  print(dictionary['label'])