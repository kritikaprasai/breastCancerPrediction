#for euclidean distance

from math import sqrt


q1 = [1,3]
p1=[2,5]
type(q1)

#print(q1.shape) #yo shape le dimension dincha since dimension array ko huncha not of list  
#tehi bhayera error aako ho hai 


euclidean_distance = sqrt((q1[0]-p1[0])**2 + (q1[1]-p1[1])**2)
print(euclidean_distance)
type(euclidean_distance)





#A Counter is a container that keeps track of how many times 
#equivalent values are added. It can be used to implement 
#the same algorithms for 
#which bag or multiset data structures are commonly used in other languages.










#=============================================================================================


import numpy as np 
from math import sqrt
import matplotlib.pyplot as plt
import warnings
import pandas as pd 
import random 
from collections import Counter
#style.use('haudeynumber')

dataset ={'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]
#for i in dataset:
#    for ii in dataset[i]:
#        [plt.scatter(ii[0],ii[1],s=100,color=i)]
#plt.scatter(new_features[0],new_features[1],s=100)
#plt.show()


def k_nearest_neighbour(data,predict,k=3):
    if len(data)>=k:
        warnings.warn('k is set to value less than total voting groups')
    distances = []
    for group in data:
        for features in data[group]:
            
#             euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
#             euclidean_distance = sqrt((features[0]-predict[0])**2 + (features[1]-predict[1])**2)
             euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
             distances.append([euclidean_distance,group])
    votes =[i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1) [0][1] / k
    print(vote_result,confidence)
    return vote_result,confidence
 
df = pd.read_csv("breast-cancer-wisconsin.data.txt")
print(df.shape)
#dataframe ko  value access garna lako 
df.iloc[40]
#dataset ko info haru ko lagi ho hai yo
#df.info()
#dataset ma bhako missing values lai mark gareko
df.describe()
#dataset ko first 20 data display garne 
df.head(20)
#replace gara ? into Nan
x = df.replace('?', np.NaN)
print(x)
x.iloc[139]
#fill missing values with mean column values
x.fillna(x.mean(), inplace=True)



#df.isnull().sum()

#replacing data 

#missing data '?' lai pahile nan le replace gara 
#data_name[‘column_name’].replace(0, np.nan, inplace= True)
#aba chai mean le unknown data lai mean value le replace gardincha 
#Age is a column name for our train data
#mean_value=train['Age'].mean()
#train['Age']=train['Age'].fillna(mean_value)
##this will replace all NaN values with the mean of the non null values
##For Median
#meadian_value=train['Age'].median()
#train['Age']=train['Age'].fillna(median_value)





df.replace('?',-99999,inplace = True)
df.drop(['id'],1,inplace=True)    
full_data = df.astype(float).values.tolist()

print(full_data[:10])
random.shuffle(full_data)

test_size = 0.2 
train_set = {2:[],4:[]}
test_set = {2:[],4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
    
for i in test_data:
    test_set[i[-1]].append(i[:-1])
correct =0 
total =0


for group in test_set:
    for data in test_set[group]:
        random_state=42
        vote,confidence = k_nearest_neighbour(train_set,data,k=3)
        if group == vote:
            correct +=1
        else:
            print(confidence)
            
        total += 1
        
        
print('Accuracy is ',correct/total)

example_measures = np.array( [1,2,1,1,1,2,2,2,1]  )
example_measures =example_measures.reshape(1,-1)
print(example_measures.shape)
prediction = k_nearest_neighbour(train_set,example_measures,k=3)

print(prediction)


if(prediction[0]==4):
    print('your prediction is  A malignant tumor is a tumor that may invade its surrounding tissue or spread around the body.',prediction)
elif(prediction[0]==2):
    print('your predictionis begnin A benign tumor is a tumor that does not invade its surrounding tissue or spread around the body.',prediction)
else:
    print('nothing')
        



#not needed 
#for i in dataset:
#    for ii in dataset[i]:
#        [plt.scatter(ii[0],ii[1],s=100,color=i)]
#plt.scatter(new_features[0],new_features[1],color=result)
#plt.show()













