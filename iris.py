import numpy as np
from numpy import genfromtxt
from sklearn import linear_model

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
Iris.csv file total number of recordes is 150 records.
trainingSet is Iris.csv file is 0 to 129 records and
testingSet is Iris.csv file is 130 to 149 records.
'''
trainingSet=file[:130]
testingSet=file[130:]

'''
Training Dataset
trainingX is to 0 to 4 columns (training feauters) and
trainingY is 5 columns (training target).
'''
trainingX=trainingSet[:,[0,1,2,3,4]]
trainingX=trainingX.astype(float)
trainingY=trainingSet[:,[5]]

'''
Testing Dataset
testingX is to 0 to 4 columns (testing feauters) and
testingY is 5 columns (testing target).
'''
testingX=testingSet[:,[0,1,2,3,4]]
testingX=testingX.astype(float)
testingY=testingSet[:,[5]]

# Implimantation of Logistic Regression
lr=linear_model.LogisticRegression()
lr.fit(trainingX,trainingY)

#Test Data : input data is 0 t0 19 because testingSet is 130 to 149 records.
print("Predict Value is "+str(lr.predict([testingX[12]])))
iname=int(testingY[12])
for var,value in dic.items():
    if value==iname:
        print("Real Iris name is "+var)
print("Accuracy : "+str(lr.score(testingX,testingY)*100))

''' 
OUTPUT :
jigar@jimmy:~$ cd Desktop/
jigar@jimmy:~/Desktop$ cd Iris/
jigar@jimmy:~/Desktop/Iris$ python iris.py
/home/jigar/anaconda2/lib/python2.7/site-packages/sklearn/utils/validation.py:578: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
Predict Value is ['2']
Real Iris name is Iris-virginica
Accuracy : 100.0
'''
