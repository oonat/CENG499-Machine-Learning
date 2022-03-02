import numpy as np


def initialize_cluster_centers(data, k):
    min_x = None
    max_x = None
    min_y = None
    max_y = None

    for point in range(data.shape[0]):
        x, y = data[point]
        min_x = x if (min_x == None or min_x > x) else min_x
        max_x = x if (max_x == None or max_x < x) else max_x
        min_y = y if (min_y == None or min_y > y) else min_y
        max_y = y if (max_y == None or max_y < y) else max_y


    centers = []
    for i in range(k):
        x_val = np.random.uniform(min_x, max_x)
        y_val = np.random.uniform(min_y, max_y)
        centers.append((x_val, y_val))

    return np.array(centers)


def assign_clusters(data, cluster_centers):
    """
    Assigns every data point to its closest (in terms of Euclidean distance) cluster center.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: An (N, ) shaped numpy array. At its index i, the index of the closest center
    resides to the ith data point.
    """
    num_example = data.shape[0]
    assign_mat = np.ndarray(shape=(num_example,), dtype=int)

    for i in range(num_example):
        dist_mat = np.linalg.norm(cluster_centers - data[i], axis=1)
        assign_mat[i] = np.argmin(dist_mat)

    return assign_mat


def calculate_cluster_centers(data, assignments, cluster_centers, k):
    """
    Calculates cluster_centers such that their squared Euclidean distance to the data assigned to
    them will be lowest.
    If none of the data points belongs to some cluster center, then assign it to its previous value.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param assignments: An (N, ) shaped numpy array with integers inside. They represent the cluster index
    every data assigned to.
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :param k: Number of clusters
    :return: A (K, D) shaped numpy array that contains the newly calculated cluster centers.
    """
    updated_centers = np.copy(cluster_centers)

    for i in range(k):
        assigned_data = data[np.where(assignments == i)]
        if assigned_data.size != 0:
            updated_centers[i] = np.sum(assigned_data, axis=0)/assigned_data.shape[0]

    return updated_centers



def kmeans(data, initial_cluster_centers):
    """
    Applies k-means algorithm.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param initial_cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: cluster_centers, objective_function
    cluster_center.shape is (K, D).
    objective function is a float. It is calculated by summing the squared euclidean distance between
    data points and their cluster centers.
    """
    cluster_num = initial_cluster_centers.shape[0]
    data_num = data.shape[0]
    cluster_centers = initial_cluster_centers

    assigned_coords = np.empty_like(data)
    objective = 0

    while True:
        assignments = assign_clusters(data, cluster_centers)
        updated_centers = calculate_cluster_centers(data, assignments, cluster_centers, cluster_num)

        for i in range(data_num):
            assigned_coords[i] = cluster_centers[assignments[i]]

        objective = np.sum((data - assigned_coords)**2)/2

        if (cluster_centers == updated_centers).all():
            break

        cluster_centers = updated_centers


    return cluster_centers, objective