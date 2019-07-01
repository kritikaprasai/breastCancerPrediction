import numpy as np
#yo sabko library buja hai
from sklearn import preprocessing,neighbors
from sklearn.model_selection import train_test_split
import pandas as pd


#data read
df= pd.read_csv('breast-cancer-wisconsin.data.txt')# yo data kaha bata read garne ho bhaneko hai ahile nothing written hai

print(df)
df.max()


#1241232,1197510,1193683............. yo id ma unknown ? value 

#missing data ko lagi
df.replace('?' ,-99999,inplace=True) 
#When inplace=True is passed, the data is renamed in place (it returns nothing)
#When inplace=False is passed (this is the default value, so isn't necessary), performs the operation and returns a copy of the object
#timro csv file ma nachine data lai faleko 
df.drop(['id'],1,inplace=True)
print(df.shape)

#yo X features ko lagi except class column
X = np.array(df.drop(['class'],1))
#yo Y label ko lagi i.e infectious or not infectious
y= np.array(df['class'])
print(y)
print(X)
print(X.shape)
print(y.shape)
print(X[698][1])
X_train ,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2 )


clf= neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)
#print(clf.fit(X_train,y_train))

from sklearn.externals import joblib

joblib.dump(clf, 'model.pkl') 
clf=joblib.load('model.pkl')
accuracy= 
clf.score(X_test,y_test)

print(accuracy)


example_measures = np.array( [1,2,1,1,1,2,2,2,1]  )
example_measures =example_measures.reshape(1,-1)
print(example_measures.shape)
prediction = clf.predict(example_measures)

print(prediction)


if(prediction[0]==4):
    print('your prediction is  A malignant tumor is a tumor that may invade its surrounding tissue or spread around the body.',prediction)
elif(prediction[0]==2):
    print('your predictionis begnin A benign tumor is a tumor that does not invade its surrounding tissue or spread around the body.',prediction)
else:
    print('nothing')
        