
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
print('Class labels :',np.unique(y))
