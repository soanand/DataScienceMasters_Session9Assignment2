# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 18:52:15 2018

@author: soanand

Problem Statement:
In this assignment students will build the random forest model after normalizing the
variable to house pricing from boston data set.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
boston = datasets.load_boston()
features = pd.DataFrame(boston.data, columns=boston.feature_names)
targets = pd.DataFrame(boston.target)


from sklearn.preprocessing import Normalizer
norm = Normalizer()
X = norm.fit_transform(features)
y = targets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)

#fitting the random forest classification to the training sets
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10, oob_score=True, random_state=0)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

#Step 6. Plotting the graph for 
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")

