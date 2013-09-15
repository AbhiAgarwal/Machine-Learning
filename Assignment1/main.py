import copy
from numpy import dot

training_data = [] # data to train, validate, and test with
validation_data = []
test_data = [] 
vocabulary_list = {} # vocabulary/word list
feature_list = [] # email feature list
weight_vector = [] # weight vector

# Calling the information from the files
def getData():
	global training_data # Edit the global varabile
	global validation_data # Edit the global varabile
	global test_data # Edit the global varabile
	# Declaration of files
	train = open('./data/spam_train.txt', 'r')
	test = open('./data/spam_test.txt', 'r')
	# Counter for 0-4000, 4000-5000
	n = 0
	# Getting data from the files
	for data in train:
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

# Getting word list out of the arrays
def rankWords():
	global vocabulary_list # Edit the global varabile
	vocabulary_list_before = {}
	n = 0
	# goes through the first 4000 words
	for i in range(0, 4000):
		# splits the word set of the training data
		word_set = (training_data[i].split())
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

# Gets single data set from the given data
def getSingleDataSet(data):
	data_set = data.split()
	counter = 0
	for data in data_set:
		if data == '1':
			del data_set[counter]
		elif data == '0':
			del data_set[counter]
		counter += 1
	return data_set

# Sorts words into order
# Returns Sorted List
def sortWords():
	return sorted(vocabulary_list.items(), key = lambda item: item[1])

# Determining if the current emails is spam or not
# Returns 1 if Spam
# Returns 0 if Not Spam
def spamornot(word_list, number):
	toBeDetermined = word_list[number]
	if toBeDetermined[:2].replace(" ", "") == "1":
		return 1
	elif toBeDetermined[:2].replace(" ", "") == "0":
		return 0

# Dot Product of Feature Vector and Weight Vector
def dotProduct(values, weight):
	weight_values = []
	for i in weight:
		weight_values.append(i[0])
	return dot(weight_values, values)

# Copies the vocabulary list and sets it as the weight vector
# and sets each weight to 0
def copyAsWeightVector():
	global weight_vector # Edit the global varabile
	weight = list(training_data)
	counter = 0
	for i in weight:
		weight[counter] = 0
		counter += 1
	weight_vector = list(weight)

# Finding position of a particular word in the Weight Vector
def findPositionInWeight(word):
	counter = 0
	for i in weight_vector:
		if i == word:
			return counter
		counter += 1

# Creates weight set
def makeWeightSet(thisset):
	weight_set = []
	singleSet = getSingleDataSet(thisset)
	count = 0
	for i in singleSet:
		thisVector = weight_vector[count]
		weight_set.append(thisVector)
		count += 1
	return weight_set

# "For each email, transform it into a feature vector x
# where the ith entry, xi, is 1 if the ith vector in the vocabulary
# occurs in the email, and 0 otherwise."
def featureWord(email_list):
	global feature_list # Edit the global varabile
	for i in range(0, 4000): # Doing this for EACH email
		word_set = getSingleDataSet(email_list[i])
		feature_list.append([])
		count = 0
		for word in word_set:
			if word in vocabulary_list:
				if word is not 0 and word is not 1:
					feature_list[i].append(1)
					count += 1
			else:
				if word is not 0 and word is not 1:
					feature_list[i].append(0)
					count += 1

# Trains a perceptron classifier using the examples
# provided to the function.
# Returns: Final classification vector, number of updates (mistakes)
# and number of passes through the data (iterations)
def perceptron_train(data):
	iterations = 0 # Number of passes through data
	mistakes = 0 # Number of mistakes/updates performed
	dataPoint = 0 # What part of the data are we at
	# traversal through the data
	while True:
		errorCounter = 0
		for row in data:
			# the word_set should be exactly mapped onto inOrNot
			word_set = getSingleDataSet(row) # The current set of words
			inOrNot = feature_list[dataPoint] # If current word it is in the word_set
			desiredOutput = spamornot(data, dataPoint) # 1 = Spam, 0 = Not Spam
			weight_set = [] # set of all weights
			for words in word_set: # Traversal through the words to check the weight
				weightForAll = makeWeightSet(words)
				weight_set.append(weightForAll)
			wFunction = dotProduct(inOrNot, weight_set) # weight function
			wFunctionNew = 0 # if incorrect the new weight function
			y = 0 # y function, to decide if spam or not spam
			if wFunction >= 0: # Checking if the Dot Product is greater or equal to zero, or less
				y = 1 # Dot product greater so equals 1
			elif wFunction < 0: # Dot product is less so equals -1
				y = -1
			if y == desiredOutput: # If correct - no change, if not then update using w = w + y(i) * f(x)
				print "correct!"
			else:
				print "update"
				# update it here (not done yet)
			dataPoint += 1 # counter for internal functions
		# if the errorCounter is 0 or more than 15 iterations have been done, then break out of the while loop
		if errorCounter == 0 or iterations >= 15:
			break
		else: # or add one iteration
			iterations += 1
	return

# The main running function
if __name__ == '__main__':
	getData() # Gets data from the files
	rankWords() # Ranks Words by how many times they appear
	featureWord(training_data) # Transform into features, input vectors
	copyAsWeightVector() # Copy training_data -> Weight Vector so they can be weighted
	perceptron_train(training_data) # Runs main perceptron algorithm