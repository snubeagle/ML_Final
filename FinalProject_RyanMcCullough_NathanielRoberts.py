#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:49:32 2020

@author: Ryan McCullough, Nathaniel Roberts
"""
# General Attributions:
# Numeric & array processing functions by NumPy
#   -- https://numpy.org
#
# Data manipulation & statistical operations by Pandas
#   -- https://pandas.pydata.org
#
# Data division & preliminary modeling with SciKit-Learn
#   -- https://scikit-learn.org
#
# Plotting & visualization tools by MatPlotLib
#   -- https://matplotlib.org
#
# Advanced machine & statistical learning methods by TensorFlow and Keras
#   -- https://www.tensorflow.org
#   -- https://keras.io
#
# Significant use was made of the official API documentation and user
# guides of each of these external packages, and the authors have our
# thanks.  Unless otherwise noted, no code was directly copied from an
# outside source, and any resemblance to such code here is unintentional.

import os
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold, train_test_split
from matplotlib import pyplot as plt


def loadData():
    ''' Read a .CSV file into memory and return it as a Pandas dataframe '''
    cwpath = "/ne15.csv"
    dpath = os.getcwd() + cwpath
    print("Loading Data ...")
    data = pd.read_csv(dpath)
    return data


def cropData(data):
        ''' Exclude data irrelevant to this operation, as well as entries
            for which relevant data is incomplete '''
        drop_cols = ["flagCa", "flagMg", "flagK", "flagNa", "flagNH4",
                     "flagNO3", "flagCl", "flagSO4", "flagBr", "valcode",
                     "invalcode"]
        data.drop(columns=drop_cols, inplace=True)
        data = data.applymap(lambda x: np.NaN if x == -9 else x)
        data = data.loc[:, ['NO3', 'SO4', 'Cl', 'NH4']]
        data = data.query('NO3 != "NaN" & SO4 != "NaN" &'
                          + 'Cl != "NaN" & NH4 != "NaN"')
        return data


def splitData(data, fold=0):
    ''' Divide data into Testing, Validation, and Training segments or,
        if a fold value is provided, into k cross-validation folds '''
    # data = cropData(data)
    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1:]
    y = np.ravel(y)
    # y = np.reshape(y, (1447,1))
    if fold > 0:
        kfolds = KFold(n_splits=fold, shuffle=True)
        folded_data = []
        for train_inds, test_inds in kfolds.split(X):
            this_fold = {"train_X": X.iloc[train_inds, :],
                            "train_y": y.iloc[train_inds, :],
                            "test_X": X.iloc[test_inds, :],
                            "test_y": y.iloc[test_inds, :]}
            folded_data.append(this_fold)
        return folded_data
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8)
    train_X, validate_X, train_y, validate_y = train_test_split(train_X,
                                                                train_y,
                                                                train_size=0.75)
    return (test_X, train_X, validate_X, train_y, test_y, validate_y)


def trainModel(data):
    ''' Construct and train the artificial neural network model '''
    (xTest, xTrain, xVal, yTrain, yTest, yVal) = splitData(data)

    model = Sequential()
    model.add(Dense(15, activation='linear', input_shape=(3,)))
    model.add(Dense(12, activation='linear'))
    model.add(Dense(9, activation='linear'))
    model.add(Dense(6, activation='linear'))
    model.add(Dense(3, activation='linear'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adagrad', loss='mean_squared_logarithmic_error')

    history = model.fit(xTrain, yTrain, epochs=10000, validation_data=(xVal, yVal))
    
    yPred = np.array(model.predict(xTest))
    eval = model.evaluate(xTest, yTest, batch_size=128)
    print("Overall loss:", eval)
    print(model.summary())
    
    '''
    for i in range(len(yPred)):
        print("Actual value: ")
        print(yTest[i][0])
        print("\n")
        print("Predicted value: ") 
        print(yPred[i])
        print("\n")'''
        

    return (model, xTest, yTest, yPred)


def fullPlot(X_data, y_true, y_pred):
    base = ['rh', 'gh', 'bh']
    elem = ['NO3', 'SO4', 'Cl']
    plt.figure(figsize=(14, 6))
    plt.suptitle("NH4 Concentrations: Observations v. Predictions")
    for n in range(3):
        ploc = 130 + (n + 1)
        plt.subplot(ploc)
        plt.xlabel(f"{elem[n]} (G/m^2)")
        plt.plot(X_data[:, n], y_true, base[n], alpha=0.3,
                 label=f"NH4 relative to {elem[n]} (observed)")
        plt.plot(X_data[:, n], y_pred, 'xk', alpha=0.3,
                 label=f"NH4 relative to {elem[n]} (predicted)")
        plt.legend()
    plt.savefig("./results.png", format='png')
    plt.show()


if __name__ == '__main__':
    data = cropData(loadData())
    (trainedModel, xTest, yTest, yPred) = trainModel(data)
    fullPlot(xTest, yTest, yPred)
