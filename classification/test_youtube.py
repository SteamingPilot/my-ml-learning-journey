# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Import our dataset
dataset = pd.read_csv("../dataset/Social_Network_Ads.csv")


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import  train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


from sklearn.linear_model import LogisticRegression

regressor = LogisticRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(accuracy_score(y_test, y_pred) )

from sklearn.model_selection import cross_val_score
c = cross_val_score(estimator=regressor, X=X, y=y, cv=100)

print(c.mean())


