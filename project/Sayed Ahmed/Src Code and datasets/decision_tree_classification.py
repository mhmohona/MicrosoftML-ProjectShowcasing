# Decision Tree Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as myplt
import pandas as pd

# Importing the dataset social network dataset
dataset = pd.read_csv('Dataset.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values #the independent variable

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training Decision Tree ML Algorithm
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the result
y_pred = classifier.predict(X_test)

# Generating Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)

# Visualising the results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
myplt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('white', 'blue')))
myplt.xlim(X1.min(), X1.max())
myplt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    myplt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('pink', 'blue'))(i), label = j)
myplt.title('Decision Tree - Training set')
myplt.xlabel('Age')
myplt.ylabel('Estimated Salary')
myplt.legend()
myplt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
myplt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('white', 'blue')))
myplt.xlim(X1.min(), X1.max())
myplt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    myplt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('pink', 'blue'))(i), label = j)
myplt.title('Decision Tree - Test set')
myplt.xlabel('Age')
myplt.ylabel('Estimated Salary')
myplt.legend()
myplt.show()