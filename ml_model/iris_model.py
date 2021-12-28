import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import pickle
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X, y)
# ---------------------------------------
fig_handle = plt.figure()
x = np.linspace(0, 2*np.pi)
y = np.sin(x)
plt.plot(x, y)

iris_list = [clf, fig_handle]

# Save the model as a pkl file
filename = 'ml_model/iris_model.pkl'
pickle.dump(iris_list, open(filename, 'wb'))
