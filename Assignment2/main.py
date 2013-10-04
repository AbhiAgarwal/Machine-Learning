from numpy import dot # Dot Function
training_data = [] # Training Data
validation_data = [] # Validation Data
test_data = [] # Test Data
weight_list = {} # Weight List
feature_list = [] # Feature Set
vocabulary_list = {} # vocabulary/word list

# Gets data from train & test
# Returns nothing
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
			test_data.append(testdata) # put data into test_data array
	return

# Splits data into many words, and removes '0' or '1'
# Returns Split & Removed of '0', '1' Data Set
def getSingleDataSet(data):
	data_set = data.split() # splits the giant array/dict
	counter = 0 # sets counter to 0
	for data in data_set: # goes through the data set 
		if data == '1': # deletes 1, cleanup
			del data_set[counter]
		elif data == '0': # deletes 0, cleanup
			del data_set[counter]
		counter += 1 # increases counter
	return data_set

# Gets list of words which occur 30 or more times
# Returns nothing
def rankWords():
	global vocabulary_list
	global weight_list
	vocabulary_list_before = {} # before you do < 30 choosings
	for i in range(0, 4000): # goes through the first 4000 words
		# splits the word set of the training data
		word_set = getSingleDataSet(training_data[i])
		for word in word_set:
			if word in vocabulary_list_before:
				# adds if the word is there
				vocabulary_list_before[word] += 1
			else:
				vocabulary_list_before[word] = 1 # if it hasn't been created
			weight_list = vocabulary_list_before # set one instance of the vocabulary list as the weight
	for i in vocabulary_list_before: # goes through all words above 30 appearances or more
		if vocabulary_list_before[i] >= 30: # puts values of above 30 into the vocabulary list
			vocabulary_list[i] = vocabulary_list_before[i]
	for i in weight_list:
		weight_list[i] = 0
	return

# Create feature set of email set (x with sub(i))
# Returns nothing
def featureWord(email_list):
	global feature_list # Edit the global varabile
	for i in range(0, 4000): # Doing this for EACH email
		word_set = getSingleDataSet(email_list[i]) # Get Single data set of the word_set
		feature_list.append([]) # Adds one instance of a multidimensional array
		for word in word_set: # traverse through each word
			if word in vocabulary_list: # If the word is in the vocabulary list
				feature_list[i].append(1) # then append '1' to the end of the Feature List
			else:
				feature_list[i].append(0) # else append '0' to the end of the Feature List
	return

# Determines if it is spam or not
# Returns 0 or 1
def spamornot(word_list, number):
	toBeDetermined = word_list[number] # checks the word to be determined
	if toBeDetermined[:2].replace(" ", "") == "1": # checks if the first "x " is 1 or 0
		return 1 # returns 1 if spam
	elif toBeDetermined[:2].replace(" ", "") == "0":
		return 0 # returns 0 if not spam

# Trains using Support Vector Machine Algorithm
# Returns final classification vector
def pegasos_svm_train(data, lambda):
	iterations = 20 # Num of passes through data
	for i in range(0, iterations): # 20 passes through data
		for i in data: # Traversal through the data
			
	return

# Tests the SVM Algorithm
def pegasos_svm_test(data, w):
	return

if __name__ == '__main__':
	getData() # Gets data from all the files
	rankWords() # Ranks Words by how many times they appear
	featureWord(training_data) # Transform into features, input vectors