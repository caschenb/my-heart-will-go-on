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

def process_data(data):
	if 'Survived' in data.columns:
		train_y = data['Survived']
		data.drop('Survived', 1, inplace=True)
	else:
		train_y = None

	# Remove useless features
	for item_to_drop in ['PassengerId', 'Name', 'Ticket', 'Cabin']:
	    data.drop(item_to_drop, 1, inplace=True)

	# Features we are going to scale and use
	print(data.columns)
	# (['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
	# Feature Scaling
	# use binarization for Age ? (split in categories ex: child, teen, ...)
	data['Age'] = preprocessing.Imputer(strategy="mean").fit_transform(
	    data['Age'].values.reshape(-1, 1))  # could try with mean
	data['Age'] = preprocessing.scale(data['Age'])
	data['Fare'] = preprocessing.Imputer(strategy="mean").fit_transform(
	    data['Fare'].values.reshape(-1, 1))
	data['Fare'] = preprocessing.scale(data['Fare'])
	data['Sex'] = pd.get_dummies(data['Sex'])
	embarked_dummies = pd.get_dummies(data['Embarked'])
	data = data.join(embarked_dummies)
	data.drop("Embarked", 1, inplace=True)

	print(data.head())

	return data, train_y

# For Kaggle evaluation
def evaluate():
	classifier = VotingClassifier(estimators=[('svc', SVC()), ('lr', LogisticRegression()),
		('rf', RandomForestClassifier()), ('gnb', GaussianNB()), ('ada', AdaBoostClassifier())], voting='hard')
	train = pd.read_csv("input/train.csv", dtype={"Age": np.float64},)
	test = pd.read_csv("input/test.csv", dtype={"Age": np.float64},)
	X, y = process_data(train)
	X_test, y_test = process_data(pd.DataFrame(test))
	classifier.fit(X,y)
	score = classifier.predict(X_test)
	print(score)
	print(test.columns)
	results = pd.DataFrame(test['PassengerId'])
	results['Survived'] = score.tolist()
	results.to_csv('results.csv', sep=',', index=False)



def test():
	# Load Data
	train = pd.read_csv("input/train.csv", dtype={"Age": np.float64},)
	print(train.head())

	train, train_y = process_data(train)

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



if __name__ == '__main__':
	evaluate()
    