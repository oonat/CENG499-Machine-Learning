from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np


train_set = np.load('hw3_material/svm/task4/train_set.npy')
train_lbs = np.load('hw3_material/svm/task4/train_labels.npy')
test_set = np.load('hw3_material/svm/task4/test_set.npy')
test_labels = np.load('hw3_material/svm/task4/test_labels.npy')

train_sample_count, train_x, train_y = train_set.shape
train_set = train_set.reshape((train_sample_count, train_x * train_y))

test_sample_count, test_x, test_y = test_set.shape
test_set = test_set.reshape((test_sample_count, test_x * test_y))



def tester(dataset, labels, balanced=None):

	clf = SVC(kernel='rbf', C=1, class_weight=balanced)
	clf.fit(dataset, labels)

	accuracy = clf.score(test_set, test_labels)
	preds = clf.predict(test_set)
	conf_matrix = confusion_matrix(test_labels, preds)

	return accuracy, conf_matrix



def oversample(minority_class, diff):

	oversampled_set = train_set
	oversampled_lbs = train_lbs
	minority_set = oversampled_set[np.where(oversampled_lbs == minority_class)]
	minority_size = minority_set.shape[0]

	for i in range(diff):
		oversampled_set = np.append(oversampled_set, [minority_set[i % minority_size]], axis=0)

	oversampled_lbs = np.append(oversampled_lbs, np.repeat(minority_class, diff))

	return oversampled_set, oversampled_lbs



def undersample(majority_class, diff):

	undersampled_set = train_set
	undersampled_lbs = train_lbs
	majority_indices = np.where(undersampled_lbs == majority_class)[0][:diff]

	undersampled_set = np.delete(undersampled_set, majority_indices, axis=0)
	undersampled_lbs = np.delete(undersampled_lbs, majority_indices)

	return undersampled_set, undersampled_lbs


def svm_task4():

	total_count = train_lbs.shape[0]
	ones_count = np.count_nonzero(train_lbs)
	zeros_count = total_count - ones_count

	if ones_count > zeros_count:
		majority_class = 1
		minority_class = 0
		diff = ones_count - zeros_count
	else:
		majority_class = 0
		minority_class = 1
		diff = zeros_count - ones_count


	oversampled_set, oversampled_lbs = oversample(minority_class, diff)
	undersampled_set, undersampled_lbs = undersample(majority_class, diff)

	unbalanced_acc, unbalanced_conf = tester(train_set, train_lbs)
	oversample_acc, oversample_conf = tester(oversampled_set, oversampled_lbs)
	undersample_acc, undersample_conf = tester(undersampled_set, undersampled_lbs)
	balanced_acc, balanced_conf = tester(train_set, train_lbs, 'balanced')

	print("-" * 8 + "IMBALANCED" + "-" * 8)
	print(f"Accuracy: {unbalanced_acc}")
	print(f"Confusion matrix: {unbalanced_conf} \n")

	print("-" * 8 + "OVERSAMPLED" + "-" * 8)
	print(f"Accuracy: {oversample_acc}")
	print(f"Confusion matrix: {oversample_conf} \n")

	print("-" * 8 + "UNDERSAMPLED" + "-" * 8)
	print(f"Accuracy: {undersample_acc}")
	print(f"Confusion matrix: {undersample_conf} \n")

	print("-" * 8 + "CLASS WEIGHT BALANCED" + "-" * 8)
	print(f"Accuracy: {balanced_acc}")
	print(f"Confusion matrix: {balanced_conf}")




svm_task4()