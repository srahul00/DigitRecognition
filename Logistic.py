import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from LogisticRegressionClassifier import LogisticRegression
import os

def logisticModel():
    print("\nLogistic Regression Classifier")
    path = os.getcwd() + "/mnist.csv"
    df = pd.read_csv(path)
    
    outputs = np.array(df['label'])
    features = np.array(df.drop(labels = ['label'], axis = 1))
    
    X_train, X_test, Y_train, Y_test = train_test_split(features, outputs, test_size = 0.25)
    
    #clf = LogisticRegression()
    clf = LogisticRegression(learningRate = 0.001, epochs = 75) 
    clf.fit(X_train, Y_train)
    acc = clf.score(X_test, Y_test)
    print(f'Accuracy : {acc}')