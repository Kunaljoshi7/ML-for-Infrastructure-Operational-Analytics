import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sensor_columns = ['sensor_{}'.format(i + 1) for i in range(50)]
metadata = ["start_MP", "end_MP", "MP", "Run"]
labels = ["label"]

def preprocessing(df):
    

    #Drop Metadata Columns for training
    df = df.drop(metadata, axis = 1)

    #Drop Nan values
    df = NAN(df)
    return df


def feature_visualize(df):
    for sensor_name in sensor_columns:
        df.plot(y = sensor_name, c='k')
        df.plot(y = sensor_name, kind = "kde")

#Check for NAN values
def NAN(df):

    df.isnull().values.any()
    print("Total NAN values:", df.isnull().sum().sum())
    print("Dataframe shape before:", df.shape)
    is_NaN = df.isnull()

    row_has_NaN = is_NaN.any(axis=1)
    rows_with_NaN = df[row_has_NaN]
    print("Rows with NAN:", rows_with_NaN)

    df = df.dropna()
    print("Dataframe shape after NAN drop:", df.shape)
    return df


#Function to visualize class imbalnaced using bar graph
def visualize_bar_graph(df):
    val_counts = df.value_counts(["label"])
    #df.groupby(['label'])
    print(val_counts)
    val_counts.plot(kind = "bar")

"""
def train_test_split(X, y, test_size, method):
    if method == "stratify":

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, stratify = y)
    
    return X_train, X_test, y_train, y_test
"""
