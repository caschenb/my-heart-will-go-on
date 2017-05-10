import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Load Data
train = pd.read_csv("input/train.csv" , dtype={"Age": np.float64},)
print(train.head())
#Feature Scaling
train["Age"][np.isnan(train["Age"])] = np.median(train["Age"])
#Plots
#plt.figure()
sns.pairplot(train[['Fare', 'Sex', 'Survived']], hue="Survived", dropna=True)
plt.show()

#Fit Algorithms


#Cross Validation
