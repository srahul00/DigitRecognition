from sklearn import model_selection, neighbors, svm
import pandas as pd
import numpy as np

df = pd.read_csv("D:/Spyder/Projects/Digit Recognizer/train.csv")
#print(df.head)

outputs = np.array(df['label'])
features = np.array(df.drop(labels = ['label'], axis = 1))

#print(outputs.head)
#print(features.head) 

X_train, X_test, y_train, y_test = model_selection.train_test_split(features, outputs, test_size = 0.2)
classifier = svm.SVC()
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)
print(f'Accuracy : {accuracy*100}') #96.98