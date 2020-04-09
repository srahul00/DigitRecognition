import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Dense
from keras.layers import Flatten
import os

def cnnModel():
    print("CNN CLassifier")
    path = os.getcwd()
    df = pd.read_csv(path + "/mnist.csv")
    Labels = df['label']
    Labels = to_categorical(Labels)
    df = df.drop(labels = ['label'],axis= 1)
    df =  df.astype('float32')
    df = df/255;
    
    
    trainX,testX,trainY,testY = train_test_split(df,Labels)
    #trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    #testX = testX.reshape((testX.shape[0], 28, 28, 1))
    trainX = np.array(trainX).reshape((trainX.shape[0], 28, 28, 1))
    testX = np.array(testX).reshape((testX.shape[0], 28, 28, 1))
    model = Sequential();
    model.add(Conv2D(64, (2,2) ,activation = 'relu', input_shape = (28,28,1)))
    model.add(AveragePooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer = 'adadelta',metrics = ['accuracy'],loss='categorical_crossentropy')
    
    model.fit(trainX,trainY,validation_data = (testX,testY),epochs = 4)