from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_classif as infoGain
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2
import pandas as pd


def normalizeNumerical(input_dataframe, numerical_features):
    """
    Normalizes numerical columns using mean and standard deviation.

    Basis:
        https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/
    Dependency:
        from sklearn.preprocessing import StandardScaler
    Input:
        input_dataframe: Pandas DataFrame
            DataFrame to be normalized
        numerical_features: list
            Numerical features to be normalized
    
    Output:
        input_dataframe: Pandas DataFrame
            Normalized dataframe based on numerical features.
    """
    
    for i in numerical_features:
        if i in input_dataframe.columns:
            scale = StandardScaler().fit(input_dataframe[[i]]) # normalizes the function for each numerical column
            input_dataframe[i] = scale.transform(input_dataframe[[i]])
    return input_dataframe


def standardizeNumerical(input_dataframe, numerical_features):
    """
    Standizes numerical columns using min max scaler.
    
    Basis:
        https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/
    Dependency:
        from sklearn.preprocessing import MinMaxScaler
    Input:
        input_dataframe: Pandas DataFrame
            DataFrame to be standardized
        numerical_features: list
            Numerical features to be normalized
    
    Output:
        input_dataframe: Pandas DataFrame
            Normalized dataframe based on numerical features.
    """
    
    for i in numerical_features:
        if i in input_dataframe.columns:
            scale = MinMaxScaler().fit(input_dataframe[[i]]) # standardizes numerical values using the MinMaxScaler
            input_dataframe[i] = scale.transform(input_dataframe[[i]])
    return input_dataframe


def corrAttrEval(input_dataframe, label_to_be_checked, ratio_included=0.5):
    """
    Performs Correlation Attribute Evaluation using the Pearson Correlation Coefficient
    
    Basis:
        https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
    Input:
        input_dataframe: Pandas DataFrame
            DataFrame to be Filtered
        label_to_be_checked: str
            Label to be used for reference
        ratio_included: float: Default = 0.5
            Minimum value of correlation to be accepted
            Value: [0, 1]
    Output:
        filtered_dataframe: Pandas DataFrame
            Filtered DataFrame
    """
    
    pearson_correlation = input_dataframe.corr() # Pandas natively has a correlation function using pearson, spearman, and kendall
    target_correlation = abs(pearson_correlation[label_to_be_checked]) # # gets all the pearson correlation: which by default the label is included
    relevant_features = target_correlation[target_correlation > ratio_included] # gets the features with correlation greater than a set ratio
    relevant_features = list(relevant_features.reset_index().iloc[:, 0]) # turns the features into a list
    filtered_dataframe = input_dataframe[relevant_features] # extracts the best features from the dataframe
    return filtered_dataframe


def infoGainAttrEval(input_dataframe, label_to_be_checked, ratio_included=0.5):
    """
    Performs Information Gain Attribute Evalutation
    Works only for discrete labels.
    
    Basis:
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html
    Dependencies:
        from sklearn.feature_selection import mutual_info_classif as infoGain
    Input:
        input_dataframe: Pandas DataFrame
            DataFrame to be Filtered
        label_to_be_checked: str
            Label to be used for reference
        ratio_included: float: Default = 0.5
            Minimum value of correlation to be accepted
            Value: [0, 1]
    Output:
        filtered_dataframe: Pandas DataFrame
            Filtered DataFrame
    """
    
    features = input_dataframe.drop(columns = [label_to_be_checked, 'Patient Identification Number']) # creates a feature dataset
    target = input_dataframe[label_to_be_checked] # creates a target dataset
    
    information = pd.DataFrame(zip(features.columns, infoGain(features, target, random_state=1))).set_index(0) # Creates a dataframe containing the information gain from each feature
    relevant_features = list(information[information[1] > ratio_included].reset_index().iloc[:, 0]) # turns the optimal features to a list
    filtered_dataframe = input_dataframe[relevant_features] # extracts the optimal features
    return filtered_dataframe.join(input_dataframe[label_to_be_checked]) # add the label column


def adaboostFeatureSelect(input_dataframe, label_to_be_checked, ratio_included=0.5):
    """
    Performs Adaboost Feature Selection
    Works only for discrete labels.
    
    Basis:
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
    Dependencies:
        from sklearn.ensemble import AdaBoostClassifier
    Input:
        input_dataframe: Pandas DataFrame
            DataFrame to be Filtered
        label_to_be_checked: str
            Label to be used for reference
        ratio_included: float: Default = 0.5
            Minimum value of correlation to be accepted
            Value: [0, 1]
    Output:
        filtered_dataframe: Pandas DataFrame
            Filtered DataFrame
    """
    
    features = input_dataframe.drop(columns = [label_to_be_checked, 'Patient Identification Number']) # creates a feature dataset
    target = input_dataframe[label_to_be_checked] # creates a target dataset

    select = AdaBoostClassifier().fit(features.values, target.values).feature_importances_ # gets the feature importance scores
    information = pd.DataFrame(zip(features.columns, select)).set_index(0) # creates a dataframe containing the feature scores
    relevant_features = list(information[information[1] > ratio_included].reset_index().iloc[:, 0]) # gets the best features
    filtered_dataframe = input_dataframe[relevant_features] # extract the best features
    return filtered_dataframe.join(input_dataframe[label_to_be_checked]) # combines the labels to the dataset


def randomForestPrep(input_dataframe, label_to_be_checked):
    """
    Performs Random Forest Feature Selection
    Works only for discrete labels.
    
    Basis:
        https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f
    Dependencies:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectFromModel
    Input:
        input_dataframe: Pandas DataFrame
            DataFrame to be Filtered
        label_to_be_checked: str
            Label to be used for reference
    Output:
        output_dataframe: Pandas DataFrame
            Filtered DataFrame
    """
    features = input_dataframe.drop(columns = [label_to_be_checked, 'Patient Identification Number']) # creates a feature dataframe
    target = input_dataframe[label_to_be_checked] # creates a target dataframe
    
    selector = SelectFromModel(RandomForestClassifier(n_estimators = 100)).fit(features.values, target.values) # Selects features from the Random Forest Classifier
    selected_features = features.columns[(selector.get_support())] # gets which features to get
    output_dataframe = input_dataframe[selected_features].join(input_dataframe[label_to_be_checked]) # extracts bets features and combines the label dataset
    return output_dataframe


def chi2PreProp(input_dataframe, label_to_be_checked, ratio_included=0.0055):
    """
    Performs Random Forest Feature Selection
    Works only for discrete labels.
    
    Basis:
        https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f
    Dependencies:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectFromModel
    Input:
        input_dataframe: Pandas DataFrame
            DataFrame to be Filtered
        label_to_be_checked: str
            Label to be used for reference
        ratio_included: float: 0.0055 i.e. 0.55%
            P-value to be compared
    Output:
        output_dataframe: Pandas DataFrame
            Filtered DataFrame
    """
    
    features = input_dataframe.drop(columns = [label_to_be_checked, 'Patient Identification Number']) # creates a feature dataframe
    target = input_dataframe[label_to_be_checked] # creates a target dataframe
    _, pValues = chi2(features.values, target.values) # computes p-value
    selected_features = features.columns[pValues< ratio_included] # gets which features to get based on a specific rato to be compared
    output_dataframe = input_dataframe[selected_features].join(input_dataframe[label_to_be_checked]) # extracts bets features and combines the label dataset
    return output_dataframe


def oneHotEncoding(input_dataframe, categorical_features):
    """
    Converts all inputted categorical features to one-hot encoding variables
    
    Input:
        input_dataframe: Pandas DataFrame
            DataFrame to be modified
        categorical_features: list
            List of categorical features found in the DataFrame
    Output:
        output_dataframe: Pandas DataFrame
            DataFrame with modified categorical values
    """
     
    for i in categorical_features:
        if i in input_dataframe.columns:
            input_dataframe = input_dataframe.join(pd.get_dummies(input_dataframe[i], prefix=i)).drop(columns=[i])
    return input_dataframe