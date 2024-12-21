
import numpy as np
from sklearn import datasets


iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
print('Class labels :',np.unique(y)) # return the three unique class labels in iris.target eventually if not it converts them to integers
# Class labels: [0 1 2]
from sklearn.model_selection import train_test_split
print(f'y dataset before: {y}')
#this method below shift the train datasets and shuffle them, so the split wil be random 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 1, stratify= y
)
print(f'y dataset after: {y}')
print('Label counts in y: ', np.bincount(y)) # this method just counts the occurrences 
# Labels counts in y: [50 50 50]
print('Labels counts in y_train: ', np.bincount(y_train))
# Labels counts in y_train: [35 35 35]
print('Labels counts in y_test: ', np.bincount(y_test))
# Labesl counts in y_test: [15 15 15]
