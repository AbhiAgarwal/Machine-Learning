import copy # copying dict/array function
from numpy import dot # dot function
import time # time (to measure)

training_data = [] # data to train the program
validation_data = [] # data to validate the training
test_data = [] # data to test the whole program

vocabulary_list = {} # vocabulary/word list
all_word_List = {} # list of all words
feature_list = [] # email feature list

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
	data_set = data.split() # splits the giant array/dict
	counter = 0 # sets counter to 0
	for data in data_set: # goes through the data set 
		if data == '1': # deletes 1
			del data_set[counter]
		elif data == '0': # deletes 0
			del data_set[counter]
		counter += 1 # increases counter
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
	toBeDetermined = word_list[number] # checks the word to be determined
	if toBeDetermined[:2].replace(" ", "") == "1": # checks if the first "x " is 1 or 0
		return 1 # returns 1 if spam
	elif toBeDetermined[:2].replace(" ", "") == "0":
		return 0 # returns 0 if not spam

# Dot Product of Feature Vector and Weight Vector
# Takes in value, and weight and returns the dot product
# of the words
def dotProduct(values, weight):
	weight_values = [] # creates a set of values
	for i in weight: # goes through all the values
		weight_values.append(i[0]) # puts the "values" set into correct order
	return dot(weight_values, values) # returns the dot product of them

# Copies the vocabulary list and sets it as the weight vector
# and sets each weight to 0
# Required as we need to adjust and get the weight of each word
def copyAsWeightVector():
	global all_word_List # Edit the global variable
	for i in all_word_List:
		all_word_List[i] = 0

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
	weight_set = [] # instance of an empty array
	singleSet = getSingleDataSet(thisset) # find the Single Set
	for i in singleSet: # Go through the set of Single words
		thisVector = all_word_List[i] # traverse through the word set
		weight_set.append(thisVector) # appends it to the end of the set
	return weight_set

# "For each email, transform it into a feature vector x
# where the ith entry, xi, is 1 if the ith vector in the vocabulary
# occurs in the email, and 0 otherwise."
def featureWord(email_list):
	global feature_list # Edit the global varabile
	for i in range(0, 4000): # Doing this for EACH email
		word_set = getSingleDataSet(email_list[i]) # Get Single data set of the word_set
		feature_list.append([]) # Adds one instance of a multidimensional array
		count = 0 # set Counter to zero
		for word in word_set: # traverse through each word
			if word in vocabulary_list: # If the word is in the vocabulary list
				if word is not 0 and word is not 1: # and the word is not '0' or '1'
					feature_list[i].append(1) # then append '1' to the end of the Feature List
					count += 1
			else:
				if word is not 0 and word is not 1:
					feature_list[i].append(0) # else append '0' to the end of the Feature List
					count += 1

# Updates toe weight set according to the new w,
# y, and set
# MOST INEFFICIENT
# 
# THINK ABOUT: http://stackoverflow.com/questions/184643/what-is-the-best-way-to-copy-a-list-in-python
def updateWeightSet(yFunction, inOrNot, word_set):
	count = 0
	for loop in word_set:
		all_word_List[loop] += (yFunction * inOrNot[count])	
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
		errorCounter = 0 # Counter for error in each iteration
		for row in data: # the word_set should be exactly mapped onto inOrNot
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
				updateWeightSet(y, inOrNot, word_set)
				mistakes += 1 # global mistakes
				errorCounter += 1 # local counter
			dataPoint += 1 # counter for internal functions
		end232 = time.time()
		if errorCounter == 0 or iterations >= 2: # if the errorCounter is 0 or more than 15 iterations have been done,
			break # then break out of the while loop
		else: # or add one iteration
			iterations += 1
			dataPoint = 0 # set dataPoint (index) back to zero
			print all_word_List
			break
	return

def perceptron_test(w, data):
	return

# The main running function
if __name__ == '__main__':
	getData() # Gets data from the files
	rankWords() # Ranks Words by how many times they appear
	featureWord(training_data) # Transform into features, input vectors
	copyAsWeightVector() # Copy training_data -> Weight Vector so they can be weighte
	start = time.time()
	perceptron_train(training_data) # Runs main perceptron algorithm
	end = time.time()
	print (end - start)