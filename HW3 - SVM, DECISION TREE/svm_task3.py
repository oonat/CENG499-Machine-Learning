from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np


train_set = np.load('hw3_material/svm/task3/train_set.npy')
train_lbs = np.load('hw3_material/svm/task3/train_labels.npy')
test_set = np.load('hw3_material/svm/task3/test_set.npy')
test_labels = np.load('hw3_material/svm/task3/test_labels.npy')

train_sample_count, train_x, train_y = train_set.shape
train_set = train_set.reshape((train_sample_count, train_x * train_y))

test_sample_count, test_x, test_y = test_set.shape
test_set = test_set.reshape((test_sample_count, test_x * test_y))

parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [0.1, 1, 10], 'gamma': [0.001, 0.01]}


def grid_search():
	clf = SVC()
	cv = GridSearchCV(clf, parameters)
	cv.fit(train_set, train_lbs)

	for i in range(18):
		param = cv.cv_results_['params'][i]
		val_acc = cv.cv_results_['mean_test_score'][i]
		print(f"Validation acc is {val_acc} for kernel = {param['kernel']} | C = {param['C']} | gamma = {param['gamma']}")

	return cv.best_params_


def svm_task3():
	best_params = grid_search()

	best_clf = SVC(kernel=best_params['kernel'], 
		C=best_params['C'],
		gamma=best_params['gamma'])

	best_clf.fit(train_set, train_lbs)
	acc_score = best_clf.score(test_set, test_labels)

	return best_params, acc_score



best_params, score = svm_task3()
print(f"Best parameters: {best_params}")
print(f"Test set accuracy for the best parameters: {score}")