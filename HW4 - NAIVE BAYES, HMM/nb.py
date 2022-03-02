import numpy as np
from math import log


def vocabulary(data):
    """
    Creates the vocabulary from the data.
    :param data: List of lists, every list inside it contains words in that sentence.
                 len(data) is the number of examples in the data.
    :return: Set of words in the data
    """

    return set([item for sublist in data for item in sublist])
    

def estimate_pi(train_labels):
    """
    Estimates the probability of every class label that occurs in train_labels.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :return: pi. pi is a dictionary. Its keys are class names and values are their probabilities.
    """
    
    pi_dict = dict()
    for label in train_labels:
        pi_dict[label] = pi_dict.get(label, 0) + 1

    for key in pi_dict:
        pi_dict[key] /= len(train_labels)

    return pi_dict



    
def estimate_theta(train_data, train_labels, vocab):
    """
    Estimates the probability of a specific word given class label using additive smoothing with smoothing constant 1.
    :param train_data: List of lists, every list inside it contains words in that sentence.
                       len(train_data) is the number of examples in the training data.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :param vocab: Set of words in the training set.
    :return: theta. theta is a dictionary of dictionaries. At the first level, the keys are the class names. At the
             second level, the keys are all the words in vocab and the values are their estimated probabilities given
             the first level class name.
    """

    theta = {}
    temp_dict = {}
    d = len(vocab)

    for i in range(len(train_labels)):
        temp_dict[train_labels[i]] = temp_dict.get(train_labels[i], []) + train_data[i]

    for label in temp_dict:
        all_count = len(temp_dict[label])
        theta[label] = {}
        for word in vocab:
            theta[label][word] = (1 + temp_dict[label].count(word)) / (d + all_count)


    return theta

    

def test(theta, pi, vocab, test_data):
    """
    Calculates the scores of a test data given a class for each class. Skips the words that are not occurring in the
    vocabulary.
    :param theta: A dictionary of dictionaries. At the first level, the keys are the class names. At the second level,
                  the keys are all of the words in vocab and the values are their estimated probabilities.
    :param pi: A dictionary. Its keys are class names and values are their probabilities.
    :param vocab: Set of words in the training set.
    :param test_data: List of lists, every list inside it contains words in that sentence.
                      len(test_data) is the number of examples in the test data.
    :return: scores, list of lists. len(scores) is the number of examples in the test set. Every inner list contains
             tuples where the first element is the score and the second element is the class name.
    """
    
    scores = []
    for example in test_data:
        tmp = []
        filtered_words = list(filter(lambda x: x in vocab, example))
        for c_i in pi:
            theta_sum = sum([log(theta[c_i][word]) for word in filtered_words])
            score = log(pi[c_i]) + theta_sum
            tmp.append((score, c_i))
        scores.append(tmp)


    return scores