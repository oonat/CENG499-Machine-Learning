import numpy as np


def single_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the single linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    data_dim = c1.shape[1]
    diff_mat = np.reshape(c1[:, np.newaxis] - c2, (-1, data_dim))

    return np.min(np.linalg.norm(diff_mat, axis=1))


def complete_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the complete linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    data_dim = c1.shape[1]
    diff_mat = np.reshape(c1[:, np.newaxis] - c2, (-1, data_dim))

    return np.max(np.linalg.norm(diff_mat, axis=1))


def average_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the average linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    data_dim = c1.shape[1]
    diff_mat = np.reshape(c1[:, np.newaxis] - c2, (-1, data_dim))

    return np.sum(np.linalg.norm(diff_mat, axis=1))/(c1.shape[0] * c2.shape[0])


def centroid_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the centroid linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """

    c1_center = np.sum(c1, axis=0)/c1.shape[0]
    c2_center = np.sum(c2, axis=0)/c2.shape[0]

    return np.linalg.norm(c1_center - c2_center)


def hac(data, criterion, stop_length):
    """
    Applies hierarchical agglomerative clustering algorithm with the given criterion on the data
    until the number of clusters reaches the stop_length.
    :param data: An (N, D) shaped numpy array containing all of the data points.
    :param criterion: A function. It can be single_linkage, complete_linkage, average_linkage, or
    centroid_linkage
    :param stop_length: An integer. The length at which the algorithm stops.
    :return: A list of numpy arrays with length stop_length. Each item in the list is a cluster
    and a (Ni, D) sized numpy array.
    """

    cluster_list = [np.expand_dims(i, axis=0) for i in data]

    while True:
        cluster_num = len(cluster_list)
        if cluster_num <= stop_length:
            break

        target_clusters = None
        min_distance = -1

        for i in range(cluster_num - 1):
            for j in range(i + 1, cluster_num):
                dist = criterion(cluster_list[i], cluster_list[j])
                if min_distance == -1 or dist < min_distance:
                    min_distance = dist
                    target_clusters = (i, j)

        t1, t2 = target_clusters
        cluster_list[t1] = np.concatenate((cluster_list[t1], cluster_list[t2]), axis=0)
        cluster_list.pop(t2)

    return cluster_list
