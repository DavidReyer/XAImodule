import shap
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, unique


from predict_time_sources import DataType, SplitMethod, feature_preparation

def loadDataset(dataset_filename, tabular_classification_filename):
    with open('datasets/' + tabular_classification_filename) as file:
        config_json = json.load(file)

    target = config_json["tabular_configuration"]["target"]["target"]
    features = config_json["tabular_configuration"]["features"].items()
    X = pd.read_csv('datasets/' + dataset_filename).drop(target, axis=1, errors='ignore')

    X = feature_preparation(X, features)

    return X, target, features


def convertObjectColumnDatatypesToCategory(X):
    # convert all object columns to categories, because autoML might only support numerical, bool and categorical features
    X[X.select_dtypes(['object']).columns] = X.select_dtypes(['object']).apply(lambda x: x.astype('category'))
    return X


def loadAutoml(filepath):
    with open('models/' + filepath, 'rb') as file:
        return pickle.load(file)


@unique
class DataType(Enum):
    DATATYPE_UNKNOW = 0
    DATATYPE_STRING = 1
    DATATYPE_INT = 2
    DATATYPE_FLOAT = 3
    DATATYPE_CATEGORY = 4
    DATATYPE_BOOLEAN = 5
    DATATYPE_DATETIME = 6
    DATATYPE_IGNORE = 7


@unique
class SplitMethod(Enum):
    SPLIT_METHOD_RANDOM = 0
    SPLIT_METHOD_END = 1


def feature_preparation(X, features):
    for column, dt in features:
        if DataType(dt) is DataType.DATATYPE_IGNORE:
            X.drop(column, axis=1, inplace=True)
        elif DataType(dt) is DataType.DATATYPE_CATEGORY:
            X[column] = X[column].astype('category')
        elif DataType(dt) is DataType.DATATYPE_BOOLEAN:
            X[column] = X[column].astype('bool')
        elif DataType(dt) is DataType.DATATYPE_INT:
            X[column] = X[column].astype('int')
        elif DataType(dt) is DataType.DATATYPE_FLOAT:
            X[column] = X[column].astype('float')
    return X