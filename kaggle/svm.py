import numpy as np
import pandas as pd


data = pd.read_csv("data/train_format.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


""" 
An efficient way to find the number of Nan's in a 2D numpy Array

Output: "i-th column" "Number of NaNs"

Note: This will also work for categorical variables.
"""

# for i in range(0, X.shape[1]):
# 	NumOfNan = 0
# 	for j in X[:, i]:
# 		if isinstance(j, float):
# 			if np.isnan(j):
# 				NumOfNan += 1
# 	print(i, NumOfNan)

# Taking Care of the missing values in the age variable:
from sklearn.impute import SimpleImputer

mean_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

mean_imputer.fit(X[:, 3:4])
X[:, 3:4] = mean_imputer.transform(X[:, 3:4])

for i in range(0, len(X[:, 8])):
	if isinstance(X[i, 8], str):
		X[i, 8] = X[i, 8][0]
	else:
		X[i, 8] = "None"


# Encoding the categorical variables:
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

le = LabelEncoder()
y = le.fit_transform(y)

le2 = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [6,8, 9])], remainder='passthrough')

X = np.array(ct.fit_transform(X))

# Splitting data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Model Train
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
# print(np.concatenate((y_pred.reshape(-1,1), y_test.reshape(-1, 1)), 1))


# Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

# Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=sc.transform(X), y=y, cv=100)

print("Accuracy: {} %".format(accuracies.mean() * 100))
print("Standard Deviation: {} %".format(accuracies.std() * 100))







