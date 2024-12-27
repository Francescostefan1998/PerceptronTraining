
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

# print(X_train)
print('--------------------------------')
# print(X_train_std)
print('----------------------------------------------------')
# print(X_test_std)

from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)


y_pred = ppn.predict(X_test_std)

# the following is calculating how many of the predictions made by my perceptron are are correct 
print('Misclassified examples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Accuracy : %.3f' % accuracy_score(y_test, y_pred))
print('Accuracy : %.3f' % ppn.score(X_test_std, y_test))

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
def plot_decision_regions(X, y, classifier, test_idx=None,
                          resolution=0.02):
    # setup marker generator and color map
    markers = ('o', 's', '^','v','<')
    colors  = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface 
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y== cl, 0],
                    y = X[y == cl, 1],
                    alpha = 0.8,
                    c=colors[idx],
                    marker = markers[idx],
                    label = f'Class {cl}',
                    edgecolor = 'black')
        
    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='none', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='Test set')
        
# X_combined_std = np.vstack((X_train_std, X_test_std))
# y_combined = np.hstack((y_train, y_test))
# plot_decision_regions(X=X_combined_std,
#                       y=y_combined,
#                       classifier=ppn,
#                       test_idx=range(105, 150))
# plt.xlabel('Petal length [standardized]')
# plt.ylabel('Petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import math
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

# z = np.arange(-7, 7, 0.1)
# sigma_z = sigmoid(z)
# plt.plot(z, sigma_z)
# plt.axvline(0.0, color='k')
# plt.ylim(-0.1, 1.1)
# plt.xlabel('z')
# plt.ylabel('$\sigma (z)$')
# plt.yticks([0.0, 0.5, 1.0])
# ax = plt.gca()
# ax.yaxis.grid(True)
# plt.tight_layout()
# plt.show()

def loss_1(z):
    print(f'loss_1 for z = {z} result: { 1 / (1 + math.exp(-z))}')
    return - np.log(sigmoid(z))

def loss_0(z):
    print(f'loss_0 for z = {z} result: { 1 -(1 / (1 + math.exp(-z)))} ')
    return - np.log(1-sigmoid(z))

z = np.arange(-10, 10, 2.5)
sigma_z = sigmoid(z)
print(sigma_z)

c1 = [loss_1(x) for x in z]
plt.plot(sigma_z, c1, label='L(w, b) if y = 1')
c0 = [loss_0(x) for x in z]
plt.plot(sigma_z, c0, linestyle = '--', label = 'L(w,b) if y = 0')
plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel('$\sigma(z)$')
plt.ylabel('L(w, b)')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# import math


# other way of defining the sigmoid function
# def sigmoid(z):
#    return 1 / (1 + math.exp(-z))   takes an input z, applies the exponential function 

# second example 
# Small set of z values
# z_values = [-2, 0, 2]

# for z in z_values:
#     sigma = sigmoid(z)
#     loss1 = loss_1(z)
#     loss0 = loss_0(z)
    
#     print(f"z = {z}:")
#     print(f"  Sigmoid(z) = {sigma}")
#     print(f"  Loss if y=1: {loss1}")
#     print(f"  Loss if y=0: {loss0}")
#     print()