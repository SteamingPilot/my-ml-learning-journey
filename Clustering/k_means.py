# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing:
## Import dataset
dataset = pd.read_csv("../dataset/Mall_Customers.csv")

# It makes sense to remove only the customer ID, but all other independent variables are important.
# Howerver, for this demonstration, in order to visualize later, we need to keep things 2D, so, we chose the
# 3-th and the 4-th column to be in the training set.
X = dataset.iloc[:, [3, 4]].values

# Note: We don't have any dependent variable.


## Split dataset into training and test set:
# There is no need for this, since we don't have any result to test to. So, we can use the whole thing for
# training.

## Feature Scaling:
# No need here.

# Building the model:

## First we need determine the number of clusters. We'll use Elbow method.
# Now Remember that we actually need k-means to calculate the WCSS.

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
	# We chose to run k=1 to 10
	kmeans = KMeans(n_clusters=i, init="k-means++", random_state=0)
	kmeans.fit(X)
	wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# I am choosing k=5 here based on the graph.

## Model:
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=0)


## Now let's make a dependent variable which will contain the predicted cluster number of a record.
y_kmeans = kmeans.fit_predict(X)


## Visualize:

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()





