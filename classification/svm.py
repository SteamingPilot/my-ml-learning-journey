import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sikit Learn Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Import Dataset
dataset = pd.read_csv("../dataset/Social_Network_Ads.csv")
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Model
classifier = SVC(kernel="linear", random_state=0)
classifier.fit(X_train, y_train)

# Prediction
y_pred = classifier.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))

# Cross Validation
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=2)
print(accuracies.mean()*100)
print(accuracies.std()*100)

