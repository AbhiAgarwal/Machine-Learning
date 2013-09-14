import sys
from collections import defaultdict
import collections
import re
import operator

# data to train, validate, test with
training_data = []
validation_data = []
test_data = []

# vocabulary/word list
vocabulary_list = {}

# Calling the information from the files
def call():
	# Declaration of files
	train = open('./data/spam_train.txt', 'r')
	test = open('./data/spam_test.txt', 'r')
	# Counter for 0-4000, 4000-5000
	n = 0
	# Getting data from the files
	for data in train:
		if data:
			# Getting 0-3999 data to training set
			if n < 4000:
				training_data.append(data)
				n += 1
			# Getting 4000-5000 data to validation set
			else:
				validation_data.append(data)
				n += 1
	# Getting test data from the files
	for testdata in test:
		# Getting 0-1000 data to test set
		if testdata:
			test_data.append(testdata)
	return

# Getting word list out of the arrays
def words():
	vocabulary_list_before = {}
	n = 0
	# goes through the first 4000 words
	for i in range(0, 4000):
		# splits the word set of the training data
		word_set = set(training_data[i].split())
		for word in word_set:
			if word in vocabulary_list_before:
				# adds if the word is there
				vocabulary_list_before[word] += 1
			else:
				vocabulary_list_before[word] = 1
	# goes through all words above 30 appearances or more
	for i in vocabulary_list_before:
		if vocabulary_list_before[i] >= 30:
			# puts values of above 30 into the vocabulary list
			vocabulary_list[i] = vocabulary_list_before[i]
	# removes '0' and '1', which are the spam/non spam filters
	del vocabulary_list['0']
	del vocabulary_list['1']

# Sorts words into order
def sortWords():
	return sorted(vocabulary_list.items(), key = lambda item: item[1])

# The main running function
if __name__ == '__main__':
	call()
	words()
	sortWords()