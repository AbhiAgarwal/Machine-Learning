from numpy import dot # Dot Function
import math
import time
training_data = [] # Training Data
validation_data = [] # Validation Data
test_data = [] # Test Data
weight_list = {} # Weight List
feature_list = [] # Feature Set
vocabulary_list = {} # vocabulary/word list

# @Gets data from train & test
# @Returns nothing
def getData():
	global training_data, validation_data, test_data # Editing the global varabile
	print "		-> Getting Test, Validation & Training Data"
	startTime = time.time() # Starts Timer
	train = open('./data/spam_train.txt', 'r') # Declaration of files
	test = open('./data/spam_test.txt', 'r') # Declaration of files
	for index, data in enumerate(train): # Getting data from the files
		if data:
			if index < 4000: # Getting 0-3999 data to training set
				training_data.append(data)
			else: # Getting 4000-5000 data to validation set
				validation_data.append(data)
	for testdata in test: # Getting test data from the files
		if testdata: # Getting 0-1000 data to test set
			test_data.append(testdata) # put data into test_data array
	print "			-> Took " + str(time.time() - startTime) + " seconds"
	return

# @Splits data into many words, and removes '0' or '1'
# @Returns Split & Removed of '0', '1' Data Set
def getSingleDataSet(data):
	data_set = data.split() # splits the giant array/dict
	for index, data in enumerate(data_set): # goes through the data set 
		if data == '1': # deletes 1, cleanup
			del data_set[index]
		elif data == '0': # deletes 0, cleanup
			del data_set[index]
	return data_set

# @Gets list of words which occur 30 or more times
# @Returns nothing
def rankWords():
	global vocabulary_list, weight_list
	print "		-> Putting words into Vocabulary & Weight List"
	startTime = time.time() # Starts Timer
	vocabulary_list_before = {} # before you do < 30 choosings
	for i in range(0, 4000): # goes through the first 4000 words
		word_set = getSingleDataSet(training_data[i]) # splits
		for word in word_set:
			if word in vocabulary_list_before:
				vocabulary_list_before[word] += 1 # adds if the word is there
			else:
				vocabulary_list_before[word] = 1 # if it hasn't been created
			weight_list = vocabulary_list_before # set one instance of the vocabulary list as the weight
	for i in vocabulary_list_before: # goes through all words above 30 appearances or more
		if vocabulary_list_before[i] >= 30: # puts values of above 30 into the vocabulary list
			vocabulary_list[i] = vocabulary_list_before[i]
	for i in weight_list:
		weight_list[i] = 0 # Initilizes it all to zero
	print "			-> Took " + str(time.time() - startTime) + " seconds"
	return

# @Create feature set of email set (x with sub(i))
# @Returns nothing
def featureWord(email_list):
	print "		-> Creating Feature Set"
	startTime = time.time() # Starts Timer
	global feature_list # Edit the global varabile
	for i in range(0, 4000): # Doing this for EACH email
		word_set = getSingleDataSet(email_list[i]) # Get Single data set of the word_set
		feature_list.append([]) # Adds one instance of a multidimensional array
		for word in word_set: # traverse through each word
			if word in vocabulary_list: # If the word is in the vocabulary list
				feature_list[i].append(1) # then append '1' to the end of the Feature List
			else:
				feature_list[i].append(0) # else append '0' to the end of the Feature List
	print "			-> Took " + str(time.time() - startTime) + " seconds"
	return

# @Determines if it is spam or not
# @Returns -1 or 1
def spamornot(word_list, number):
	toBeDetermined = word_list[number] # checks the word to be determined
	if toBeDetermined[:2].replace(" ", "") == "1": # checks if the first "x " is 1 or 0
		return 1 # returns 1 if spam
	elif toBeDetermined[:2].replace(" ", "") == "0":
		return 0 # returns 0 if not spam

# @Trains using Support Vector Machine Algorithm
# @Returns final classification vector
def pegasos_svm_train(data, lambd):
	global weight_list
	print "		-> Training using the Pegasos Algorithm"
	startTime = time.time() # Starts Timer

	u = [0 for i in range(0, weight_list.__len__())]
	iterations = 20 # Num of passes through data
	t = 0 # Element value
	nt = 0 # n subscript t value

	for i in range(0, iterations): # 20 passes through data
		loss = 0
		for index, email in enumerate(feature_list): # Traversal through the Feature List
			t += 1
			nt = ((float(1))/(t * lambd))
			yj = 1 if (spamornot(data, index) == 0) else -1
			if(yj * dot(email, weight_list)):
				print "x"
			#if (yj * dot(wt, xt) > 1):
			#	u = (1 - (nt * lambd) * wt + nt*yj*xj )
	print "			-> Took " + str(time.time() - startTime) + " seconds"
	return

# @Tests the SVM Algorithm
# @Returns nothing
def pegasos_svm_test(data, w):
	return

# @Main function
# @Returns nothing
if __name__ == '__main__':
	print "-> Starting Process"
	startTime = time.time() # Starts Timer
	getData() # Gets data from all the files
	rankWords() # Ranks Words by how many times they appear
	featureWord(training_data) # Transform into features, input vectors
	lambd = math.pow(2,-5)
	print "-> Whole Algorithm: Took " + str(time.time() - startTime) + " seconds"