import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

'''
Iris.csv file in total number of 150 data in Iris Dataset. 50 records is Iris-setosa, next 50 records is Iris-virginica and next 50 records is Iris-versicolor.
'''
file=genfromtxt("Iris.csv",delimiter=",",dtype="str",skip_header=1)

# Species columns is String to Intger like Iris-setosa is 0,Iris-versicolor is 1 and Iris-virginica is 2
dic={}
count=0
for val in file:
    if val[5] not in dic:
        dic[val[5]]=count
        count+=1     
for val in file:
    val[5]=dic[val[5]]

'''
Training Dataset
trainingX is to 0 to 4 columns (training feauters) and
trainingY is 5 columns (training target).
'''    
trainingX=file[:,[0,1,2,3,4]]
trainingX=trainingX.astype(float)
trainingY=file[:,[5]]

# Data split in two part test-data set and train-dataset randomly
training_X,testing_X,tranining_Y,testing_Y=train_test_split(trainingX,trainingY,random_state=0)

# Implimantation of Logistic Regression
lr=linear_model.LogisticRegression()
lr.fit(trainingX,trainingY)

#Test Data : input data is 0 t0 37 because randomly 38 records test-data
print("Predict Value is "+str(lr.predict([testing_X[12]])))
iname=int(testing_Y[12])
for var,value in dic.items():
    if value==iname:
        print("Real Iris name is "+var)
print("Accuracy : "+str(lr.score(testing_X,testing_Y)*100))

'''
OUTPUT :
jigar@jimmy:~/Desktop/Iris$ python iris1.py
/home/jigar/anaconda2/lib/python2.7/site-packages/sklearn/utils/validation.py:578: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
Predict Value is ['1']
Real Iris name is Iris-versicolor
Accuracy : 89.47368421052632
'''
