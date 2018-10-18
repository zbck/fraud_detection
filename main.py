from sklearn.model_selection import StratifiedKFold
from Feature_selection import Feature_selection
from Random_forest import Random_forest
from Metrics import Metrics

# TODO : strore the different metrics for each loop
# TODO : Correct warnings
# TODO : Add logs

if __name__ == '__main__':
   
    # Load data and label 
    filepath_data = 'creditcard.csv'
    feature_select = Feature_selection(filepath_data)
    data = feature_select.get_data()
    label = feature_select.get_label()
    
    # Cross validation
    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(data, label):

        # Split data into train and test sets
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
		
        # Balance the class sample
        X_train_bal, y_train_bal = feature_select.balance_class_smote(X_train, y_train)

        # Train the Random forest
        rdm_forest = Random_forest(X_train_bal, X_test, y_train_bal, y_test)
        # Display metrics
        rdm_forest.display_metrics()
        	

	
	

