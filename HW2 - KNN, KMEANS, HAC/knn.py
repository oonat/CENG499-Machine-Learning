import numpy as np


def calculate_distances(train_data, test_instance, distance_metric):
    """
    Calculates Manhattan (L1) / Euclidean (L2) distances between test_instance and every train instance.
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data.
    :param test_instance: A (D, ) shaped numpy array.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: An (N, ) shaped numpy array that contains distances.
    """
    diff_matrix = train_data - test_instance

    if distance_metric == 'L1':
        return np.sum(np.abs(diff_matrix), axis=1)
    else:
        return np.linalg.norm(diff_matrix, axis=1)


def majority_voting(distances, labels, k):
    """
    Applies majority voting. If there are more then one major class, returns the smallest label.
    :param distances: An (N, ) shaped numpy array that contains distances
    :param labels: An (N, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :return: An integer. The label of the majority class.
    """
    idx_nearest = np.argpartition(distances, k)[:k]
    labels_nearest = labels[idx_nearest]

    unique_labels, counts = np.unique(labels_nearest, return_counts=True)
    major_labels = unique_labels[np.argmax(counts)]

    # return the smallest label
    return np.min(major_labels)


def knn(train_data, train_labels, test_data, test_labels, k, distance_metric):
    """
    Calculates accuracy of knn on test data using train_data.
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param train_labels: An (N, ) shaped numpy array that contains labels
    :param test_data: An (M, D) shaped numpy array where M is the number of examples
    and D is the dimension of the data
    :param test_labels: An (M, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: A float. The calculated accuracy.
    """

    pred_list = []
    for tdata in test_data:
        distances = calculate_distances(train_data, tdata, distance_metric)
        pred_list.append(majority_voting(distances, train_labels, k))

    pred_list = np.array(pred_list)
    return np.sum(np.equal(pred_list, test_labels))/test_labels.shape[0]


def split_train_and_validation(whole_train_data, whole_train_labels, validation_index, k_fold):
    """
    Splits training dataset into k and returns the validation_indexth one as the
    validation set and others as the training set. You can assume k_fold divides N.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param validation_index: An integer. 0 <= validation_index < k_fold. Specifies which fold
    will be assigned as validation set.
    :param k_fold: The number of groups that the whole_train_data will be divided into.
    :return: train_data, train_labels, validation_data, validation_labels
    train_data.shape is (N-N/k_fold, D).
    train_labels.shape is (N-N/k_fold, ).
    validation_data.shape is (N/k_fold, D).
    validation_labels.shape is (N/k_fold, ).
    """
    split_data = np.vsplit(whole_train_data, k_fold)
    split_labels = np.split(whole_train_labels, k_fold)
    val_data = split_data[validation_index]
    val_labels = split_labels[validation_index]

    train_data = np.vstack(np.delete(split_data, validation_index, axis=0))
    train_labels = np.hstack(np.delete(split_labels, validation_index, axis=0))

    return train_data, train_labels, val_data, val_labels


def cross_validation(whole_train_data, whole_train_labels, k_fold, k, distance_metric):
    """
    Applies k_fold cross-validation and averages the calculated accuracies.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param k_fold: An integer.
    :param k: An integer. The number of nearest neighbor to be selected.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: A float. Average accuracy calculated.
    """
    acc_list = []
    for i in range(k_fold):
        train_data, train_labels, val_data, val_labels = \
            split_train_and_validation(whole_train_data, whole_train_labels, i, k_fold)
            
        acc_list.append(knn(train_data, train_labels, val_data, val_labels, k, distance_metric))

    return sum(acc_list)/k_fold