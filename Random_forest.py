import csv
import numpy as np

from Metrics import Metrics
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class Random_forest:


	FOREST = RandomForestClassifier(n_estimators=5, random_state=0)

	"""	
	def __init__(self, data_filepath, label_filepath):
		# Data set and the associated labels are stored
		self.DATA, self.LABELS = self._load_data(data_filepath,
												label_filepath)

	def _load_data(self, data_filepath, label_filepath):
		'''Load the data from .npy files
		'''
		data = np.load(data_filepath)
		label = np.load(label_filepath)	
		return data, label
	"""
	def __init__(self, X_train, X_test, y_train, y_test):
		self.X_TRAIN = X_train
		self.X_TEST = X_test
		self.Y_TRAIN = y_train
		self.Y_TEST = y_test

	def	rdm_forest_classifier(self):
		''' Train the random forest classifier
			and display the metrics result
		''' 
		X_train, X_test, y_train, y_test = train_test_split(self.DATA,
															self.LABELS,
															test_size=0.33)
		self.FOREST.fit(X_train, y_train.ravel())

		print('Score on the trainning set : ', self.FOREST.score(X_train, y_train))
		print('Score on the testing set : ', self.FOREST.score(X_test, y_test))
		
		y_pred = np.asarray(self.FOREST.predict(X_test), dtype=int)
		y_pred_proba = np.asarray(self.FOREST.predict_proba(X_test), dtype=float)
		y_test = np.asarray(y_test, dtype=int)
		
		self.METRICS = Metrics(y_pred, y_test, y_pred_proba)

	def display_metrics(self):
		'''	Display the following metrics:
				- Presicion
				- Recall
				- F1 score
				- AUC (Area under the curve)
			And plot the ROC curve
		'''
		metrics = self.METRICS

		precisison = metrics.get_precision()
		recall = metrics.get_recall()
		fscore = metrics.get_fscore()
		auc_score = metrics.get_auc()
		
		#print(classification_report(y_test, y_pred))		
		#print(confusion_matrix(y_test, y_pred))		
		print('Precision : ', precisison)
		print('Recall : ', recall)
		print('F1 score : ', fscore)
		print('AUC (Area under the curve) : ', auc_score)

		metrics.plot_roc()

if __name__=='__main__':
	
	data_filepath = 'train_data_clean.npy'
	label_filepath = 'train_label_clean.npy'
	random_forest = Random_forest(data_filepath, label_filepath)
	random_forest.rdm_forest_classifier()
	random_forest.display_metrics()
