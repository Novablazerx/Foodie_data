# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 00:25:12 2020

@author: rohan
"""

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

df = pd.read_csv('2020-XTern-DS.csv')
df = df[['Restaurant', 'Latitude', 'Longitude', 'Cuisines', 'Average_Cost', 'Minimum_Order', 'Rating', 'Votes', 'Reviews', 'Cook_Time']]

df2 = df[['Cuisines','Average_Cost', 'Rating']]
df2['Average_Cost'] = df2['Average_Cost'].str.replace(",", "")
df2['Average_Cost'] = df2['Average_Cost'].str.replace("$", "").astype(float)

df2 = df2.dropna()

print(df2['Average_Cost'].head())

X = df2[['Average_Cost']]
y = df2['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_predict))


plt.figure(figsize=(25,10))
a = plot_tree(model)



df3 = df[['Average_Cost', 'Rating', 'Cook_Time']]


df3 = df3.dropna()
df3['Average_Cost'] = df3['Average_Cost'].str.replace(",", "")
df3['Average_Cost'] = df3['Average_Cost'].str.replace("$", "").astype(float)
df3['Cook_Time'] = df3['Cook_Time'].str.replace("minutes", "").astype(int)

X = df3[['Average_Cost']]
y = df3['Cook_Time']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_predict))


plt.figure(figsize=(25,10))
a = plot_tree(model)

#Observation1 : Restaurants rated with a higher average cost have higher ratings
#Observation2: More votes seem to have higher reviews"

#Observation3: Cooking time is not correlated with the average price of the restaurant."
#Observation4: Most restaurants have average price under $25.
