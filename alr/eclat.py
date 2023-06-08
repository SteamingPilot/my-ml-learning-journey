# Import libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd


# Data Preprocessing:
## Import dataset
dataset = pd.read_csv("../dataset/Market_Basket_Optimization.csv", header=None)

# No need for X, y dataset like the other ones.

# Our apriori function expects a list of lists, where the inner list is a transaction.
# Each item in the inner list must be strings
transactions = []
for i in range(0, 7501):
	transactions.append([str(dataset.values[i, j]) for j in range(0, 20) ])

## Taking care of missing data:
# N/A

## Encoding Categorical data:
# N/A

## Split dataset into training and test set:
# We will use the whole dataset to build the model.

## Feature Scaling:
# N/A

# Building the model:

## Model:
# Training
from apyori import apriori
rules = apriori(transactions = transactions, min_support=0.003, min_confidence=0.2, min_lift=3,
                min_length=2, max_length=2)


results = list(rules)

# Printing Format
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))

resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])
print(resultsinDataFrame.nlargest(n=10, columns="Support"))





