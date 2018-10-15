from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

class Metrics:

	def __init__(self, y_pred, y_test, y_pred_proba):
		self.Y_PRED = y_pred
		self.Y_TEST = y_test
		self.Y_PRED_PROBA = y_pred_proba
		self._cal_metrics()

	def _cal_metrics(self):
		''' Calculate the metrics:
				- Presicion
				- Recall
				- F1 score
				- Area Under the Curve (AUC)
		'''
		y_test = self.Y_TEST
		y_pred = self.Y_PRED
		y_pred_proba = self.Y_PRED_PROBA

		self.PRECISION = precision_score(y_test, y_pred)
		self.RECALL = recall_score(y_test, y_pred)
		self.FSCORE = f1_score(y_test, y_pred)
		self.AUC = roc_auc_score(y_test, y_pred_proba[:,1])

	def plot_roc(self):
		''' Calculate and display the ROC curve
		'''
		y_test = self.Y_TEST
		y_pred_proba = self.Y_PRED_PROBA
		
		# Calculate the ROC
		fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1], pos_label=1)

		#Display
		plt.clf()
		plt.plot(fpr, tpr)
		plt.xlabel('FPR')
		plt.ylabel('TPR')
		plt.title('ROC curve')
		plt.show()


	def get_precision(self):
		''' Return the precision : tp/(tp + fp)
			How well positive exemple truly are positive
		'''
		return self.PRECISION
			
	def get_recall(self):
		''' Return the precision : tp/(tp + fn)
			Fraction of positives examples correcly labelled 
		'''
		return self.RECALL
	
	def get_fscore(self):
		''' Return the F1 score : 2PR/(P + R)
			Help to choose the best algorithm
		'''
		return self.FSCORE	

	def get_auc(self):
		''' Return AUC (Area Under the Curve)
		'''
		return self.AUC
