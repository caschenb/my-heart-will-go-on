import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load Data
train = pd.read_csv("input/train.csv", dtype={"Age": np.float64},)
print(train.head())
train_y = train['Survived']

# Remove useless features
for item_to_drop in ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin']:
    train.drop(item_to_drop, 1, inplace=True)

# Plots
# plt.figure()
#sns.pairplot(train[['Fare', 'Sex', 'Survived']], hue="Survived", dropna=True)
# plt.show()

# Features we are going to scale and use
print(train.columns)
# (['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# Feature Scaling
# use binarization for Age ? (split in categories ex: child, teen, ...)
train['Age'] = preprocessing.Imputer(strategy="median").fit_transform(
    train['Age'].values.reshape(-1, 1))  # could try with mean
train['Age'] = preprocessing.scale(train['Age'])
train['Fare'] = preprocessing.scale(train['Fare'])
train['Sex'] = pd.get_dummies(train['Sex'])
embarked_dummies = pd.get_dummies(train['Embarked'])
train = train.join(embarked_dummies)
train.drop("Embarked", 1, inplace=True)

print(train.head())

# Cross Validation
algos = [SVC(probability=True), LogisticRegression(), RandomForestClassifier(), AdaBoostClassifier()]
for clf in algos:
	score = cross_val_score(clf, train, train_y, cv=5).mean()
	print(score)
    
#Ensemble method : combination of severals classifiers    
eclf_hard = VotingClassifier(estimators=[('svc', SVC()), ('lr', LogisticRegression()),
	('rf', RandomForestClassifier()), ('gnb', GaussianNB()), ('ada', AdaBoostClassifier())], voting='hard')

score = cross_val_score(eclf_hard, train, train_y, cv=5).mean()
print("ensemble", score)