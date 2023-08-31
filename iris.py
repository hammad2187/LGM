# #####Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

###### Loading the Dataset######3
dataset=pd.read_csv(r"C:\Users\Hammad\Downloads\Iris.csv")
print(dataset.keys())
print(dataset.head(5))

modified_dataset=dataset.drop(columns=['Id'])
print(modified_dataset)
modified_dataset.describe()
# ##pip install scikit-learn

Ebl = LabelEncoder()
modified_dataset['Species']=Ebl.fit_transform(modified_dataset['Species'])
print('transformed dataset\n',modified_dataset)

########################Split Data into Training and Testing############

X=modified_dataset.drop(columns=['Species'])
Y=modified_dataset['Species']
print('training Data\n',X)
print('Testing Data\n',Y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.50)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# ######## Model training######
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
print(f'{knn}Training Complete....')

# ############### Model Testing################
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#
predictions = knn.predict(x_test)
#
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))
#
#
# ##############Model Prediction#########################

newData=np.array([[5.1,3.5,1.4,0.2]])
newD1=np.array([[5, 2.9, 1, 0.2]])
#
print('Deminsion of New Data',newD1.shape)
prediction = knn.predict(newD1)
print((prediction))
print((dataset['Species'][prediction]))