import copy # copying dict/array function
from numpy import dot # dot function
import time # time (to measure)

training_data = [] # data to train the program
validation_data = [] # data to validate the training
test_data = [] # data to test the whole program

def getData():
	global training_data # Edit the global varabile
	global validation_data # Edit the global varabile
	global test_data # Edit the global varabile
	train = open('./data/spam_train.txt', 'r') # Declaration of files
	test = open('./data/spam_test.txt', 'r') # Declaration of files
	n = 0 # Counter for 0-4000, 4000-5000
	for data in train: # Getting data from the files
		if data:
			if n < 4000: # Getting 0-3999 data to training set
				training_data.append(data)
				n += 1
			else: # Getting 4000-5000 data to validation set
				validation_data.append(data)
				n += 1
	for testdata in test: # Getting test data from the files
		if testdata: # Getting 0-1000 data to test set
			test_data.append(testdata)
	return

if __name__ == '__main__':
	getData() # Gets data from all the files