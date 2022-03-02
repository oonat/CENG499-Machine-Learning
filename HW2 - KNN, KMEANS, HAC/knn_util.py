import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from knn import cross_validation

train_set = np.load('hw2_material/knn/train_set.npy')
train_labels = np.load('hw2_material/knn/train_labels.npy')
test_set = np.load('hw2_material/knn/test_set.npy')
test_labels = np.load('hw2_material/knn/test_labels.npy')


acc_list_L1 = []
acc_list_L2 = []

for k in range(1, 180):
	acc_list_L1.append(cross_validation(train_set, train_labels, 10, k, 'L1'))
	acc_list_L2.append(cross_validation(train_set, train_labels, 10, k, 'L2'))



def plot_graph(acc_list, metric):

	k_range = range(1, 180)
	fig, ax = plt.subplots()
	ax.plot(k_range, acc_list)
	ax.set_ylabel("Accuracy")
	ax.set_xlabel("k")
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))

	fig.savefig("KNN_"+ metric + "_PLOT.png")
	plt.close(fig)


plot_graph(acc_list_L1, 'L1')
plot_graph(acc_list_L2, 'L2')


# Test set accuracies for the best k values
best_k_L1 = acc_list_L1.index(max(acc_list_L1))
best_k_L2 = acc_list_L2.index(max(acc_list_L2))


test_acc_L1 = cross_validation(test_set, test_labels, 10, best_k_L1, 'L1')
test_acc_L2 = cross_validation(test_set, test_labels, 10, best_k_L2, 'L2')

print(f'Test set accuracy for k = {best_k_L1}  distance metric = L1 is {test_acc_L1}')
print(f'Test set accuracy for k = {best_k_L2}  distance metric = L2 is {test_acc_L2}')