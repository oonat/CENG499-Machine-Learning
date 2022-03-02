import numpy as np
from dt import *
import graphviz



class Node:

    def __init__(self, left, right, bucket, val, attr):

        self.left = left
        self.right = right
        self.bucket = bucket
        self.val = val
        self.attr = attr





def id3(data, labels, method, prepruning):

    bucket = [np.count_nonzero(labels == 0), np.count_nonzero(labels == 1), np.count_nonzero(labels == 2)]

    splits = []

    for i in range(4):
        splits.append(calculate_split_values(data, labels, 3, i, method))

    split_val = None 
    attr = None

    if method == 'info_gain':
        max_val = None

        for i in range(4):
            if(splits[i][:, 1].size == 0):
                continue

            index = np.argmax(splits[i][:, 1])
            val = splits[i][index][1]
            if max_val is None or val > max_val:
                max_val = val 
                split_val = splits[i][index][0]
                attr = i

    else:
        min_val = None

        for i in range(4):
            if(splits[i][:, 1].size == 0):
                continue

            index = np.argmin(splits[i][:, 1])
            val = splits[i][index][1]
            if min_val is None or val < min_val:
                min_val = val 
                split_val = splits[i][index][0]
                attr = i

    root = Node(None, None, bucket, split_val, attr)

    if bucket.count(0) < 2:


        left_indices = np.where(data[:, attr] < split_val)[0]
        right_indices = np.where(data[:, attr] >= split_val)[0]

        left_vals = data[left_indices]
        left_labels = labels[left_indices]
        right_vals = data[right_indices]
        right_labels = labels[right_indices]


        if prepruning:
            deg_list = [0,2.71,4.61,6.25,7.78,9.24,10.6,12,13.4,14.7,16,17.3,18.5,19.8,21.1,22.3]
            left_bucket = [np.count_nonzero(left_labels == 0), np.count_nonzero(left_labels == 1), np.count_nonzero(left_labels == 2)]
            right_bucket = [np.count_nonzero(right_labels == 0), np.count_nonzero(right_labels == 1), np.count_nonzero(right_labels == 2)]

            chi_val, deg_freedom = chi_squared_test(left_bucket, right_bucket)

            if(chi_val > deg_list[deg_freedom]):
                root.left = id3(left_vals, left_labels, method, prepruning)
                root.right = id3(right_vals, right_labels, method, prepruning)

        else:

            root.left = id3(left_vals, left_labels, method, prepruning)
            root.right = id3(right_vals, right_labels, method, prepruning)

    return root







def print_tree(w, root):

    if root.left is None and root.right is None:
        return

    w.node(str(id(root.left)), f"{root.left.attr} | {root.left.val} | {root.left.bucket}")
    w.node(str(id(root.right)), f"{root.right.attr} | {root.right.val} | {root.right.bucket}")
    w.edge(str(id(root)), str(id(root.left)), '<')
    w.edge(str(id(root)), str(id(root.right)), '>=')

    print_tree(w, root.left)
    print_tree(w, root.right)



def draw(root):
    w = graphviz.Digraph('wide')
    w.node(str(id(root)), f"{root.attr} | {root.val} | {root.bucket}")
    print_tree(w, root)
    w.view()






def test_sample(sample, root):

    if root.left is None and root.right is None:
        return np.argmax(root.bucket)
    else:
        if(sample[root.attr] >= root.val):
            return test_sample(sample, root.right)
        else:
            return test_sample(sample, root.left)

            
        
def calculate_accuracy(data, labels, root):
    counter = 0
    result_list = []

    for sample in data:
        result_list.append(test_sample(sample, root))

    for i in range(len(result_list)):
        if result_list[i] == labels[i]:
            counter += 1

    return counter / len(result_list)








train_set = np.load('hw3_material/dt/train_set.npy')
train_labels = np.load('hw3_material/dt/train_labels.npy')
test_set = np.load('hw3_material/dt/test_set.npy')
test_labels = np.load('hw3_material/dt/test_labels.npy')

root = id3(train_set, train_labels, 'info_gain', False)

draw(root)

acc = calculate_accuracy(test_set, test_labels, root)
print(f"Test set accuracy: {acc}")