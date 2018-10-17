import csv
import numpy as np

from Metrics import Metrics
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class Random_forest:
    
    FOREST = RandomForestClassifier(n_estimators=10, random_state=0)

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_TRAIN = X_train
        self.X_TEST = X_test
        self.Y_TRAIN = y_train
        self.Y_TEST = y_test
        self._rdm_forest_classifier()

    def	_rdm_forest_classifier(self):
        ''' Train the random forest classifier
            and display the metrics result
        '''
        
        self.FOREST.fit(self.X_TRAIN, self.Y_TRAIN.ravel())

        print('Score on the trainning set : ', self.FOREST.score(self.X_TRAIN, self.Y_TRAIN))
        print('Score on the testing set : ', self.FOREST.score(self.X_TEST, self.Y_TEST))
		
        y_pred = np.asarray(self.FOREST.predict(self.X_TEST), dtype=int)
        y_pred_proba = np.asarray(self.FOREST.predict_proba(self.X_TEST), dtype=float)
        y_test = np.asarray(self.Y_TEST, dtype=int)
		
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
