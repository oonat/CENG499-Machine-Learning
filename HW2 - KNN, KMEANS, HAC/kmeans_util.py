import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from kmeans import *

dataset1 = np.load('hw2_material/kmeans/dataset1.npy')
dataset2 = np.load('hw2_material/kmeans/dataset2.npy')
dataset3 = np.load('hw2_material/kmeans/dataset3.npy')
dataset4 = np.load('hw2_material/kmeans/dataset4.npy')


def restart_kmeans(data, k):
	min_obj = None

	for i in range(10):
		initial_centers = initialize_cluster_centers(data, k)
		new_centers, objective = kmeans(data, initial_centers)
		if min_obj == None or min_obj > objective:
			min_obj = objective

	return min_obj



def plot_k_versus_obj(data, dataset_index):

	obj_list = [restart_kmeans(data, k) for k in range(1, 11)]
	
	k_range = range(1, 11)
	fig, ax = plt.subplots()
	ax.plot(k_range, obj_list)
	ax.set_ylabel("Objective function value")
	ax.set_xlabel("k")
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))

	fig.savefig("Dataset"+ str(dataset_index) + "_kmeans_k_versus_obj.png")
	plt.close(fig)


def plot_clusters(data, k, dataset_index):

	initial_centers = initialize_cluster_centers(data, k)
	new_centers, objective = kmeans(data, initial_centers)
	assignments = assign_clusters(data, new_centers).tolist()

	plt.scatter([x[0] for x in data], [x[1] for x in data], c=assignments)
	plt.scatter([x[0] for x in new_centers], [x[1] for x in new_centers], c='r', marker='P')
	plt.savefig("Dataset"+ str(dataset_index) + "_cluster_plot_k_" + str(k) + '.png')
	plt.clf()


def plot():
	plot_k_versus_obj(dataset1, 1)
	plot_k_versus_obj(dataset2, 2)
	plot_k_versus_obj(dataset3, 3)
	plot_k_versus_obj(dataset4, 4)



# Draw k versus objective function plots to determine the best k values using the elbow method
plot()

# For the best k values draw cluster plots
plot_clusters(dataset1, 2, 1)
plot_clusters(dataset2, 3, 2)
plot_clusters(dataset3, 4, 3)
plot_clusters(dataset4, 4, 4)