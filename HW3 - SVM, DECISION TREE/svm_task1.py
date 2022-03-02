from sklearn.svm import SVC
import numpy as np
from draw_svm import draw_svm


train_set = np.load('hw3_material/svm/task1/train_set.npy')
train_lbs = np.load('hw3_material/svm/task1/train_labels.npy')
c_list = [0.01, 0.1, 1, 10, 100]

x1_min, x1_max, x2_min, x2_max = train_set[:, 0].min(), train_set[:, 0].max(), train_set[:, 1].min(), train_set[:, 1].max()


def svm_task1(c):
	clf = SVC(kernel='linear', C=c)
	clf.fit(train_set, train_lbs)
	draw_svm(clf, train_set, train_lbs, x1_min, x1_max, x2_min, x2_max, f'svm1_{c}.png')


for c in c_list:
	svm_task1(c)