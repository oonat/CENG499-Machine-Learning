from sklearn.svm import SVC
import numpy as np
from draw_svm import draw_svm


train_set = np.load('hw3_material/svm/task2/train_set.npy')
train_lbs = np.load('hw3_material/svm/task2/train_labels.npy')
kernel_list = ['linear', 'rbf', 'poly', 'sigmoid']

x1_min, x1_max, x2_min, x2_max = train_set[:, 0].min(), train_set[:, 0].max(), train_set[:, 1].min(), train_set[:, 1].max()


def svm_task2(kernel):
	clf = SVC(kernel=kernel, C=1)
	clf.fit(train_set, train_lbs)
	draw_svm(clf, train_set, train_lbs, x1_min, x1_max, x2_min, x2_max, f'svm2_{kernel}.png')


for ker in kernel_list:
	svm_task2(ker)