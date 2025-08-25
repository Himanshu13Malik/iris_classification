"""
Iris Dataset Classification
----------------------------
This script applies multiple ML algorithms (Logistic Regression, KNN, Decision Tree and SVC.)
on the famous Iris dataset (because i had to start somewhere), evaluates them using confusion matrices,
and visualizes the results with heatmaps.
"""



from sklearn.datasets import load_iris 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


iris = load_iris()

x = iris.data
y = iris.target

# x_train, x_test = train_test_split(x, test_size=0.2, random_state=42)
# y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)
'''can also be written as -'''
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=13)

'''
I could've used a loop to iterate between different algorithms but since this is my first time,
and I'm still learning these functions, I decided to code them individually.
'''


#using LogisticRegressios
from sklearn.linear_model import LogisticRegression as logreg

model = logreg (max_iter=200)
model.fit(x_train, y_train)

y_pred=model.predict(x_test)


from sklearn.metrics import accuracy_score
print("\nAccuracy of LogisticRegression: ", accuracy_score(y_pred,y_test))


from sklearn.metrics import confusion_matrix as cm



def show_graph(h):   #h is the cunfusion matrix that we get after testing each algorithm
    sns.heatmap(h, annot=True, fmt="d", cmap="Greens", 
                xticklabels=["Setosa", "Versicolor", "Virginica"], 
                yticklabels=["Setosa", "Versicolor", "Virginica"])


    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Heatmap")
    plt.show()


#using DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier as dtc
mod1 = dtc()
mod1.fit(x_train, y_train)
y_pred = mod1.predict(x_test)
print("\nAccuracy of decision tree is: ", accuracy_score(y_test, y_pred))
h = cm(y_pred, y_test)
print(h)
show_graph(h)


#using SVC
from sklearn.svm import SVC
mod2 = SVC(kernel = "linear")
mod2.fit(x_train,y_train)
y_pred = mod2.predict(x_test)
print("\nAccuracy of SVC Linear is: ", accuracy_score(y_test, y_pred))
h = cm(y_pred, y_test)
print(h)
show_graph(h)

mod3 = SVC(kernel = "rbf")
mod3.fit(x_train,y_train)
y_pred = mod3.predict(x_test)
print("\nAccuracy of SVC rbf is: ", accuracy_score(y_test, y_pred))
h = cm(y_pred, y_test)
print(h)
show_graph(h)


#using KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier as knc
knn = knc(n_neighbors=5)
knn.fit(x_train , y_train)
y_pred = knn.predict(x_test)
print("\nAccuracy of KNN is: ", accuracy_score(y_test, y_pred))
h = cm(y_pred, y_test)
print(h)
show_graph(h)
