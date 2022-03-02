from nb import *



def filter_special_chars(word_list):
	""" For every word in word_list, this function filters the 
	special characters placed in the word using the isalnum function, 
	which returns true if the alphanumeric, and false otherwise """

	cleaned = []

	for i in range(len(word_list)):
		filtered = ''.join(c for c in word_list[i] if c.isalnum())
		if filtered != '':
			cleaned.append(filtered)

	return cleaned



def read_set(fname, remove_special):

	line_list = []
	with open(fname) as f:
		for line in f:
			words = line.rstrip().split(" ")
			if remove_special:
				words = filter_special_chars(words)

			line_list.append(words)

	return line_list



def read_labels(fname):

	with open(fname) as f:
		line_list = [line.rstrip() for line in f]

	return line_list



def calculate_acc(scores, test_labels):

	true_count = 0
	test_set_size = len(test_labels)

	for i in range(test_set_size):
		predicted_label = max(scores[i], key=lambda x:x[0])[1]
		if predicted_label == test_labels[i]:
			true_count += 1

	return true_count / test_set_size



def run_test(remove_special):

	train_set = read_set('nb_data/train_set.txt', remove_special=remove_special)
	train_labels = read_labels('nb_data/train_labels.txt')
	test_set = read_set('nb_data/test_set.txt', remove_special=remove_special)
	test_labels = read_labels('nb_data/test_labels.txt')
	

	vocab = vocabulary(train_set)
	pi = estimate_pi(train_labels)
	theta = estimate_theta(train_set, train_labels, vocab)

	scores = test(theta, pi, vocab, test_set)


	print(f"Test Set Accuracy: {calculate_acc(scores, test_labels)}")




if __name__ == "__main__":

	run_test(remove_special=True)