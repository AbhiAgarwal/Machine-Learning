import copy
from numpy import dot
import time

training_data = [] # data to train the program
validation_data = [] # data to validate the training
test_data = [] # data to test the whole program
vocabulary_list = {} # vocabulary/word list
all_word_List = {} # list of all words
feature_list = [] # email feature list
weight_vector = [] # weight vector

timeStart = 0
timeEnd = 0

def startTime():
	global timeStart
	timeStart = time.time()

def endTime(function_name):
	global timeStart
	global timeEnd
	timeEnd = time.time()
	print function_name, (timeEnd - timeStart)

# Calling the information from the files
# Gets the training, validation, and test data
# Splits train data (5000) into 2 -> 
# 1000 for validation, and 4000 for training
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
# Gets all the words that are most frequent
# and puts them into a seperate dict: vocabulary_list
# The words have to occur 30 or more times to be in this
# dictionary.
def rankWords():
	global vocabulary_list # Edit the global varabile
	global all_word_List
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
	all_word_List = vocabulary_list_before
	for i in vocabulary_list_before:
		if vocabulary_list_before[i] >= 30:
			# puts values of above 30 into the vocabulary list
			vocabulary_list[i] = vocabulary_list_before[i]
	# removes '0' and '1', which are the spam/non spam filters
	del vocabulary_list['0']
	del vocabulary_list['1']

# Gets single data set from the given data
# Usually require a split data set for all words
# 'Bat', 'hello', rather than "bat hello". So the split
# helps to resolve that very easily, and removing the 1
# and 0 so they don't come into the calculation for ranking.
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

# Sorts words into order (if required)
# Returns Sorted List
def sortWords():
	sortedList = sorted(vocabulary_list.items(), key = lambda item: item[1])
	return sortedList

# Determining if the current emails is spam or not
# Just checks the first two characters of the particular email
# Returns 1 if Spam
# Returns 0 if Not Spam
def spamornot(word_list, number):
	toBeDetermined = word_list[number]
	if toBeDetermined[:2].replace(" ", "") == "1":
		return 1
	elif toBeDetermined[:2].replace(" ", "") == "0":
		return 0

# Dot Product of Feature Vector and Weight Vector
# Takes in value, and weight and returns the dot product
# of the words
def dotProduct(values, weight):
	weight_values = []
	for i in weight:
		weight_values.append(i[0])
	return dot(weight_values, values)

# Copies the vocabulary list and sets it as the weight vector
# and sets each weight to 0
# Required as we need to adjust and get the weight of each word
def copyAsWeightVector():
	global weight_vector # Edit the global variable
	weight = list(all_word_List)
	counter = 0
	for i in weight:
		weight[counter] = 0
		counter += 1
	weight_vector = list(weight)

# Finding position of a particular word in the Weight Vector
# We need to find the word in the weight vector so this would
# allow us to quickly find it and alter / get the value
# when we need too.
def findPositionInWeight(word):
	counter = 0
	for i in all_word_List:
		train = getSingleDataSet(i)
		for i in train:
			if i == word:
				return counter
		counter += 1

# Creates weight set
# Creates a weight set for a particular word
# and returns that set, you've to traverse through it
# in order to get the set of all the words
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

# Updates toe weight set according to the new w,
# y, and set
# MOST INEFFICIENT
def updateWeightSet(yFunction, inOrNot, word_set):
	count = 0
	for loop in word_set:
		positionOfWordInWeightVector = all_word_List.keys().index(loop) # Position of word in Vector
		currentWeightVector = weight_vector[positionOfWordInWeightVector] # the current weight vector
		weight_vector[positionOfWordInWeightVector] += (yFunction * inOrNot[count])
		count += 1
	count = 0

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
			weight_set = [] # set of all weights for the current row
			for words in word_set: # Traversal through the words to check the weight
				weightForAll = makeWeightSet(words)
				weight_set.append(weightForAll) # adding weight to the row
			wFunction = dotProduct(inOrNot, weight_set) # weight function
			y = 0 # y function, to decide if spam or not spam
			if wFunction >= 0: # Checking if the Dot Product is greater or equal to zero, or less
				y = 1 # Dot product greater so equals 1
			elif wFunction < 0: # Dot product is less so equals -1
				y = -1
			if y == desiredOutput:
				hi = 0
			else: # if not then update using w = w + y(i) * f(x)
				# print "update" # we have to update each weight of each word according to the equation
				updateWeightSet(y, inOrNot, word_set)
				# update it here (not done yet)
			dataPoint += 1 # counter for internal functions
		# if the errorCounter is 0 or more than 15 iterations have been done, then break out of the while loop
		if errorCounter == 0 or iterations >= 15:
			break
		else: # or add one iteration
			iterations += 1
			print "iteration"
	return

def perceptron_test(w, data):
	return

# The main running function
if __name__ == '__main__':
	getData() # Gets data from the files
	rankWords() # Ranks Words by how many times they appear
	featureWord(training_data) # Transform into features, input vectors
	copyAsWeightVector() # Copy training_data -> Weight Vector so they can be weighted
	perceptron_train(training_data) # Runs main perceptron algorithm