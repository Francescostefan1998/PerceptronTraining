
import numpy as np
from sklearn import datasets


iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
print('Class labels :',np.unique(y)) # return the three unique class labels in iris.target eventually if not it converts them to integers
# Class labels: [0 1 2]
from sklearn.model_selection import train_test_split

#this method below shift the train datasets and shuffle them, so the split wil be random 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 1, stratify= y
)

# Print y_train and y_test
# print("y_train:", y_train)
# print("y_test:", y_test)

# print( "X_train:", X_train)
# print("X_test:", X_test)

# print('Label counts in y: ', np.bincount(y)) # this method just counts the occurrences 
# Labels counts in y: [50 50 50]
# print('Labels counts in y_train: ', np.bincount(y_train))
# Labels counts in y_train: [35 35 35]
# print('Labels counts in y_test: ', np.bincount(y_test))
# Labesl counts in y_test: [15 15 15]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)

# what the transform does? 
# basically for each X data it will calculate his scaled value
# the calculation is (X - mean) / standard deviation

# it is importand when I scaled the test data to use the mean and standard deviation of the training data
X_test_std = sc.transform(X_test)

print(X_train)
print('--------------------------------')
print(X_train_std)
print('----------------------------------------------------')
print(X_test_std)

from sklearn.linear_model import Perceptron
ppn = Perceptron(eta=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
