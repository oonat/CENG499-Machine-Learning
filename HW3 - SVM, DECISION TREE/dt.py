import math
import numpy as np

def entropy(bucket):
    """
    Calculates the entropy.
    :param bucket: A list of size num_classes. bucket[i] is the number of
    examples that belong to class i.
    :return: A float. Calculated entropy.
    """
    total_examples = sum(bucket)
    if total_examples == 0:
        return 0

    entropy = 0.

    for cl in bucket:
        if cl == 0:
            continue

        p_i = cl/total_examples
        entropy -= p_i * math.log2(p_i)

    return entropy


def info_gain(parent_bucket, left_bucket, right_bucket):
    """
    Calculates the information gain. A bucket is a list of size num_classes.
    bucket[i] is the number of examples that belong to class i.
    :param parent_bucket: Bucket belonging to the parent node. It contains the
    number of examples that belong to each class before the split.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float. Calculated information gain.
    """

    p_entro = entropy(parent_bucket)
    l_entro = entropy(left_bucket)
    r_entro = entropy(right_bucket)

    total_examples = sum(parent_bucket)
    l_ratio = sum(left_bucket) / total_examples
    r_ratio = sum(right_bucket) / total_examples

    return p_entro - r_ratio * r_entro - l_ratio * l_entro


def gini(bucket):
    """
    Calculates the gini index.
    :param bucket: A list of size num_classes. bucket[i] is the number of
    examples that belong to class i.
    :return: A float. Calculated gini index.
    """

    if sum(bucket) == 0:
        return 1

    ratio = sum([i**2 for i in bucket]) / (sum(bucket)**2)

    return 1 - ratio


def avg_gini_index(left_bucket, right_bucket):
    """
    Calculates the average gini index. A bucket is a list of size num_classes.
    bucket[i] is the number of examples that belong to class i.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float. Calculated average gini index.
    """

    l_size = sum(left_bucket)
    r_size = sum(right_bucket)
    l_ratio = l_size / (l_size + r_size)
    r_ratio = r_size / (l_size + r_size)

    return gini(left_bucket) * l_ratio + gini(right_bucket) * r_ratio


def calculate_split_values(data, labels, num_classes, attr_index, heuristic_name):
    """
    For every possible values to split the data for the attribute indexed by
    attribute_index, it divides the data into buckets and calculates the values
    returned by the heuristic function named heuristic_name. The split values
    should be the average of the closest 2 values. For example, if the data has
    2.1 and 2.2 in it consecutively for the values of attribute index by attr_index,
    then one of the split values should be 2.15.
    :param data: An (N, M) shaped numpy array. N is the number of examples in the
    current node. M is the dimensionality of the data. It contains the values for
    every attribute for every example.
    :param labels: An (N, ) shaped numpy array. It contains the class values in
    it. For every value, 0 <= value < num_classes.
    :param num_classes: An integer. The number of classes in the dataset.
    :param attr_index: An integer. The index of the attribute that is going to
    be used for the splitting operation. This integer indexs the second dimension
    of the data numpy array.
    :param heuristic_name: The name of the heuristic function. It should either be
    'info_gain' of 'avg_gini_index' for this homework.
    :return: An (L, 2) shaped numpy array. L is the number of split values. The
    first column is the split values and the second column contains the calculated
    heuristic values for their splits.
    """

    attr_list = list(data[:, attr_index])
    sorted_attr = sorted(attr_list)

    split_vals = []
    heuristic_vals = []

    for i in range(len(sorted_attr)-1):
        split_val = (sorted_attr[i] + sorted_attr[i + 1]) / 2
        split_vals.append(split_val)

        left_labels = [0] * num_classes
        right_labels = [0] * num_classes

        for j in range(data.shape[0]):
            if data[j][attr_index] < split_val:
                left_labels[labels[j]] += 1
            else:
                right_labels[labels[j]] += 1

        if heuristic_name == 'info_gain':
            parent_labels = [x + y for x, y in zip(left_labels, right_labels)]
            heuristic_vals.append(info_gain(parent_labels, left_labels, right_labels))
        else:
            heuristic_vals.append(avg_gini_index(left_labels, right_labels))

    return np.array([split_vals, heuristic_vals]).T





def chi_squared_test(left_bucket, right_bucket):
    """
    Calculates chi squared value and degree of freedom between the selected attribute
    and the class attribute. A bucket is a list of size num_classes. bucket[i] is the
    number of examples that belong to class i.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float and and integer. Chi squared value and degree of freedom.
    """
    table = np.array([left_bucket, right_bucket])
    n_l = sum(left_bucket)
    n_r = sum(right_bucket)
    total = sum(left_bucket) + sum(right_bucket)

    p = [(n_l / total), (n_r / total)]

    expected_vals = np.empty((2, len(left_bucket)))

    for i in range(2):
        for j in range(len(left_bucket)):
            expected_vals[i][j] = (table[0][j] + table[1][j]) * p[i]

    chi_val = 0

    for i in range(2):
        for j in range(len(left_bucket)):
            if expected_vals[i][j] == 0:
                continue
            chi_val += (table[i][j] - expected_vals[i][j]) ** 2 / expected_vals[i][j]


    return chi_val, np.count_nonzero(np.sum(table, axis=0))-1


