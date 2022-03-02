import numpy as np
import matplotlib.pyplot as plt
from hac import *

dataset1 = np.load('hw2_material/hac/dataset1.npy')
dataset2 = np.load('hw2_material/hac/dataset2.npy')
dataset3 = np.load('hw2_material/hac/dataset3.npy')
dataset4 = np.load('hw2_material/hac/dataset4.npy')

criterion_list = [single_linkage, complete_linkage, average_linkage, centroid_linkage]



def plot_hac(data, stop_val, dataset_index):

	for criterion in criterion_list:
		hac_clusters = hac(data, criterion, stop_val)

		for cluster in hac_clusters:
		    plt.scatter([x[0] for x in cluster], [x[1] for x in cluster])
		plt.savefig("Dataset"+ str(dataset_index) + "_hac_" + criterion.__name__ + '.png')
		plt.clf()




plot_hac(dataset1, 2, 1)
plot_hac(dataset2, 2, 2)
plot_hac(dataset3, 2, 3)
plot_hac(dataset4, 4, 4)