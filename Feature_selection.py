import csv
import numpy as np

from pathlib import Path
from collections import Counter
from imblearn.over_sampling import SMOTE


class Feature_selection:
	'''This class is used to select only features
		of samples written in a csv file.
	'''

	EXTENTION = '.csv'
	
	NEW_ARRAY = []
	NEW_DATA_ARRAY = []
	NEW_LABEL_ARRAY = []
	
	def __init__(self, filepath_data, output_data_file=None, output_label_file=None): 
		self.FILEPATH_DATA = filepath_data

		# If the file has a correct extention,open and read it
		if self._check_file():
			self._open_read_file()
			self._split_data_labels()
			self._reshape_array()
			
			# Save data and labels
			if output_data_file != None:
				self._array2file(output_data_file,
									len(self.FRAUD_FEATURES) - 1,
									self.NEW_DATA_ARRAY)
			if output_label_file != None:
				self._array2file(output_label_file,
									1,
									self.NEW_LABEL_ARRAY)

	def _check_file(self):
		'''Check if the file is a .csv
		'''
		if Path(self.FILEPATH_DATA).suffix == self.EXTENTION:
			return True
		else:
			return False			

	def _open_read_file(self):
		'''Open and read csv files.
			The spamreader will be used to read the rows of
			the csv file
		'''
		self.FILE = open(self.FILEPATH_DATA,"r")
		self.SPAMREADER = csv.reader(self.FILE)
		self.FRAUD_FEATURES = next(self.SPAMREADER) 
	
	def _split_data_labels(self):
		'''Split the data and associated labels into
			two differents arrays
		'''
		class_index = self.FRAUD_FEATURES.index('Class')
		for row in self.SPAMREADER:	
			self.NEW_DATA_ARRAY.append(np.array(row[0:len(row)-1]))
			self.NEW_LABEL_ARRAY.append(np.array(row[class_index]))
				
	def _array2file(self, output_file, nb_parameters, array):
		''' Write an array into a csv file
		'''
		np.save(output_file, np.reshape(np.array(array),
										(-1, nb_parameters)))
	def _reshape_array(self):
		''' Reshape the array into : nb_samples*nb_features
		'''
		self.DATA = np.reshape(np.array(self.NEW_DATA_ARRAY),
								(-1, len(self.FRAUD_FEATURES) - 1))
		
		self.LABEL = np.reshape(np.array(self.NEW_LABEL_ARRAY),
								(-1, 1))
								
	def balance_class_smote(self, X_train, y_train):
		'''Use SMOTE algorithm to balance classes
			return X_train_bal : balanced array of samples
			return Y_train_bal : balanced array of classes
		'''
		sm = SMOTE(random_state=42)
		X_train = np.asarray(X_train, dtype=float)
		y_train = np.asarray(y_train, dtype=int)
		print('type :', type(X_train))
		print(X_train)
		X_train_bal, y_train_bal = sm.fit_sample(X_train, y_train)
		return X_train_bal, y_train_bal
		
			
	def count_class_sample(self, labels):
		'''Display the number of sample in each class
		'''
		labels = np.reshape(np.asarray(labels, dtype=int), len(labels), 1)
		print('Class dataset shape : ', Counter(labels))

	def get_data(self):
		'''Numpy array containing the data
		'''
		return self.DATA

	def get_label(self):
		'''Numpy array containing the label
		'''
		return self.LABEL

if __name__=='__main__':

	filepath_data = 'creditcard.csv'
	#output_data_file = 'train_data_clean'
	#output_label_file = 'train_label_clean'
	param_selec = Feature_selection(filepath_data)
	param_selec.count_class_samp(param_selec.NEW_LABEL_ARRAY)
