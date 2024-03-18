import pandas as pd
import numpy as np
import os

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,accuracy_score
from sklearn.model_selection import train_test_split

import argparse

def get_data():
    try:
        URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        
        # Read data as dataframe
        df = pd.read_csv(URL,sep=";")
        return df
    except Exception as e:
        raise e

def evaluate(y_true,y_pred):
    """mae = mean_absolute_error(y_true,y_pred)
    mse = mean_squared_error(y_true,y_pred)
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    r2 = r2_score(y_true,y_pred)

    return mae,mse,rmse,r2"""

    accuracy = accuracy_score(y_true,y_pred)
    return accuracy

def main(n_estimators,max_depth):
    df = get_data()

    # Train Test Split
    train, test = train_test_split(df,random_state=215)
    x_train = train.drop("quality",axis=1)
    x_test = test.drop(["quality"],axis=1)

    y_train = train[["quality"]]
    y_test = test[["quality"]]

    # Training Model
    ''' lr = ElasticNet()
    lr.fit(x_train,y_train)
    pred = lr.predict(x_test)'''

    rf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
    rf.fit(x_train,y_train)
    pred = rf.predict(x_test)

    ## Model Evalauation
    """mae,mse,rmse,r2 = evaluate(y_test,pred)

    print(f"Mean Aboslute Error: {mae}, Mean Squared Error: {mse}, Root Mean Squared Error: {rmse}, R-Squared: {r2}")"""
    
    accuracy = evaluate(y_test,pred)

    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--n_estimators","-n",default=50, type=int)
    args.add_argument("--max_depth","-m", default=5, type=int)

    parse_args = args.parse_args()

    try:
        main(n_estimators=parse_args.n_estimators,max_depth=parse_args.max_depth)
    except Exception as e:
        raise e