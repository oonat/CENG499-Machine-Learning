import numpy as np


def forward(A, B, pi, O):
	"""
	Calculates the probability of an observation sequence O given the model(A, B, pi).
	:param A: state transition probabilities (NxN)
	:param B: observation probabilites (NxM)
	:param pi: initial state probabilities (N)
	:param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
		N is the number of states,
		M is the number of possible observations, and
		T is the sequence length.
	:return: The probability of the observation sequence and the calculated alphas in the Trellis diagram with shape
			 (N, T) which should be a numpy array.
	"""
	sequence_length = O.shape[0]
	state_count = A.shape[0]
	table = np.zeros((state_count, sequence_length))

	for state in range(state_count):
		table[state][0] = pi[state] * B[state][O[0]]

	for seq in range(1, sequence_length):
		for state in range(state_count):
			for q in range(state_count):
				table[state][seq] += table[q][seq - 1] * A[q][state]
			table[state][seq] *= B[state][O[seq]]

	return np.sum(table, axis=0)[-1], table


def viterbi(A, B, pi, O):
	"""
	Calculates the most likely state sequence given model(A, B, pi) and observation sequence.
	:param A: state transition probabilities (NxN)
	:param B: observation probabilites (NxM)
	:param pi: initial state probabilities(N)
	:param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
		N is the number of states,
		M is the number of possible observations, and
		T is the sequence length.
	:return: The most likely state sequence with shape (T,) and the calculated deltas in the Trellis diagram with shape
			 (N, T). They should be numpy arrays.
	"""

	sequence_length = O.shape[0]
	state_count = A.shape[0]
	table = np.empty((state_count, sequence_length))
	best_ind = np.empty((state_count, sequence_length), dtype=np.int)

	tmp = np.empty(state_count)

	for state in range(state_count):
		table[state][0] = pi[state] * B[state][O[0]]

	for seq in range(1, sequence_length):
		for state in range(state_count):
			for q in range(state_count):
				tmp[q] = table[q][seq - 1] * A[q][state]
			table[state][seq] = B[state][O[seq]] * np.max(tmp)
			best_ind[state][seq] = np.argmax(tmp)

	best_state = np.argmax(table[:, -1])
	path = np.empty(sequence_length)

	for i in range(sequence_length-1, -1, -1):
		path[i] = best_state
		best_state = best_ind[best_state][i]

	return path, table