import math, time, sys
all_data = [] # Training + Validation Data
training_data = [] # Training Data
validation_data = [] # Validation Data
test_data = [] # Test Data
weight_list = {} # Weight List
feature_list = [] # Feature Set
vocabulary_list = {} # vocabulary/word list
train_vector_list = [] # Vector for Training
validate_vector_list = [] # Vector for Validation
weights_pegasos = [] # THIS ONE WE USE FOR PEGASOS WEIGHT 

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
			all_data.append(data)
			if index < 4000: # Getting 0-3999 data to training set
				training_data.append(data)
			else: # Getting 4000-5000 data to validation set
				validation_data.append(data)
	for testdata in test: # Getting test data from the files
		if testdata: # Getting 0-1000 data to test set
			test_data.append(testdata) # put data into test_data array
	print "			-> Took " + str(time.time() - startTime) + " seconds"
	print "			-> There are " + str(len(all_data)) + " words in all the Data Set"
	print "			-> There are " + str(len(training_data)) + " words in the Training Data Set"
	print "			-> There are " + str(len(validation_data)) + " words in the Validation Data Set"
	print "			-> There are " + str(len(test_data)) + " words in the Test Data Set"
	return

# @Splits data into many words, and removes '0' or '1'
# @Returns Split & Removed of '0', '1' Data Set
def getSingleDataSet(data):
	if "-all" in sys.argv:
		print "Calling getSingleDataSet"
	data_set = data.split() # splits the giant array/dict
	for index, data in enumerate(data_set): # goes through the data set 
		if data == '1': # deletes 1, cleanup
			del data_set[index]
		elif data == '0': # deletes 0, cleanup
			del data_set[index]
	return data_set

# @Gets list of words which occur 30 or more times
# @Returns nothing
def rankWords(data, dataset):
	global vocabulary_list, weight_list
	print "		-> Putting words into Vocabulary & Weight List"
	print "		-> Dataset: " + dataset
	startTime = time.time() # Starts Timer
	vocabulary_list_before = {} # before you do < 30 choosings
	for i in range(0, len(data)): # goes through the first 4000 words
		word_set = getSingleDataSet(data[i]) # splits
		for word in word_set:
			if word in vocabulary_list_before:
				vocabulary_list_before[word] += 1 # adds if the word is there
			else:
				vocabulary_list_before[word] = 1 # if it hasn't been created
			weight_list = vocabulary_list_before # set one instance of the vocabulary list as the weight
	print "		-> Removing words occuring less than 30 times"
	for i in vocabulary_list_before: # goes through all words above 30 appearances or more
		if vocabulary_list_before[i] >= 30: # puts values of above 30 into the vocabulary list
			vocabulary_list[i] = vocabulary_list_before[i]
	for i in weight_list:
		weight_list[i] = 0 # Initilizes it all to zero
	print "			-> Took " + str(time.time() - startTime) + " seconds"
	print "			-> There are " + str(len(vocabulary_list)) + " words in the Vocabulary List"
	return

# @Create feature set of email set (x with sub(i))
# @Returns nothing
def featureWord(email_list, dataset):
	global feature_list # Edit the global varabile
	print "		-> Creating Feature Set"
	print "		-> Dataset: " + dataset
	startTime = time.time() # Starts Timer
	for i, email in enumerate(email_list): # Doing this for EACH email
		word_set = getSingleDataSet(email_list[i]) # Get Single data set of the word_set
		feature_list.append([]) # Adds one instance of a multidimensional array
		for word in word_set: # traverse through each word
			if word in vocabulary_list: # If the word is in the vocabulary list
				feature_list[i].append(1) # then append '1' to the end of the Feature List
			else:
				feature_list[i].append(0) # else append '0' to the end of the Feature List
	print "			-> Took " + str(time.time() - startTime) + " seconds"
	print "			-> There are " + str(len(feature_list)) + " words in the feature list"
	return

# @Determines if it is spam or not
# @Returns -1 or 1
def spamornot(word_list, number):
	if "-all" in sys.argv:
		print "Calling spamornot"
	toBeDetermined = word_list[number] # checks the word to be determined
	if toBeDetermined[:2].replace(" ", "") == "1": # checks if the first "x " is 1 or 0
		return 1 # returns 1 if spam
	elif toBeDetermined[:2].replace(" ", "") == "0":
		return 0 # returns 0 if not spam

# Splits vector training, and validation into 4000:1000 split
def createVector():
	if "-all" in sys.argv:
		print "Calling createVector"
	global validate_vector_list, train_vector_list
	train_vector_list = feature_list[0:4000]
	validate_vector_list = feature_list[4000:5000]

def vectorAdd(vecOne, vecTwo):
	if "-all" in sys.argv:
		print "Calling vectorAdd"
	vec = []
	for one, two in zip(vecOne, vecTwo):
		vec.append(one + two)
	return vec

def vectorMult(scalarQ, vecAr):
	if "-all" in sys.argv:
		print "Calling vectorMult"
	for index, vector in enumerate(vecAr):
		vecAr[index] *= scalarQ
	return vecAr

def dotFunction(vecOne, vecTwo):
	if "-all" in sys.argv:
		print "Calling dotFunction"
	return sum(one * two for one, two in zip(vecOne, vecTwo))

def vectorMagnitude(vector):
	if "-all" in sys.argv:
		print "Calling vectorMagnitude"
	div = float(0)
	for values in vector:
		div += float(math.pow(values, 2))
	magnitude = float(math.sqrt(div))
	return 0.001 if (div == 0.0) else magnitude

def evaluateFunc(weight, lambd, two):
	if "-all" in sys.argv:
		print "Calling evaluateFunc"
	obj = ((lambd / 2) * math.pow(vectorMagnitude(weight), 2)) + two
	return obj

# @Trains using Support Vector Machine Algorithm
# @Returns final classification vector
def pegasos_svm_train(vector, data, lambd, iteration):
	global weights_pegasos
	print "		-> Training using the Pegasos Algorithm"
	startTime = time.time() # Starts Timer
	iterations = 1 # Num of passes through data
	weights_pegasos = [0 for i in range(0, len(vector))] # Initilize the Weight
	u = [0 for i in range(0, len(vector))] # Initlize the "u"
	t = 0 # Element value
	nt = 0 # n subscript t value
	for i in range(0, iteration): # 20 passes through data
		overone = 0 # > 1
		underone = 0 # < 1
		evalution = 0 # eval
		for index, vec in enumerate(vector): # Traversal through the Feature List
			t += 1
			nt = ((float(1))/(t * lambd))
			yj = 1 if (spamornot(data, index) == 0) else -1
			if((yj * dotFunction(weights_pegasos, vec)) < 1):
				u = vectorAdd(vectorMult((1 - (nt * lambd)), weights_pegasos), vectorMult((nt * yj), vec))
				underone += 1
			else:
				u = vectorMult((1 - (nt * lambd)), weights_pegasos)
				overone += 1
			val = min(1, ((1 / (math.sqrt(lambd))) / (vectorMagnitude(u))))
			tempo = vectorMult(val, u)
			weights_pegasos = tempo[:]
			evalution += max(0, (1 - (nt * dotFunction(weights_pegasos, vec))))
			del u[:] #reset
		obj_val = evaluateFunc(weights_pegasos, lambd, evalution)
		print "			-> Lower Than One: " + str(underone) + " Over One: " + str(overone)
		print "			-> Number: " + str(iterations) + " Value: " + str(obj_val)
		iterations += 1
	print "			-> Took " + str(time.time() - startTime) + " seconds"
	return weights_pegasos

def checkValue(vector, weight):
	if "-all" in sys.argv:
		print "Calling checkValue"
	result = 1 if (dotFunction(vector, weight) > 0) else -1
	return result

# @Tests the SVM Algorithm
# @Returns nothing
def pegasos_svm_test(weight, feature, data):
	print "		-> Starting Test"
	print "		-> Dataset: " + "Test Data, using " + str(len(feature))
	startTime = time.time() # Starts Timer
	errors = 0

	for index, vec in enumerate(feature):
		yj = 1 if (spamornot(data, index) == 0) else -1
		result = checkValue(vec, weight)
		if result == yj:
			continue
		else:
			errors += 1

	notRight = (float(errors) / (float(len(feature))))
	print "			-> Error: " + str(errors) + ", out of: " + str(len(feature)) + ". Has " + str(notRight) + "% errors"
	print "			-> Took " + str(time.time() - startTime) + " seconds"	
	return

# @Main function
# @Returns nothing
if __name__ == '__main__':
	if "-run" in sys.argv:
		print "Calling: " + str(sys.argv) + " arguments."
		print "-> Starting Process"
		# Initialize
		startTime = time.time() # Starts Timer
		getData() # Gets data from all the files
		rankWords(training_data, "Training Data") # Ranks Words by how many times they appear
		rankWords(validation_data, "Validation Data") # Ranks Words by how many times they appear
		featureWord(training_data, "Training Data") # Transform into features, input vectors
		featureWord(validation_data, "Validation Data") # Transform into features, input vectors
		createVector()
		# Run Pegasos
		lambd = math.pow(2,-5)
		weight = pegasos_svm_train(train_vector_list, all_data, lambd, 20)
		pegasos_svm_test(weight, validate_vector_list, validation_data)
		print "-> Whole Algorithm: Took " + str(time.time() - startTime) + " seconds"
	elif "-test" in sys.argv:
		print "heh"
	else:
		print "	-run: to run training"
		print "	-test: to run testing"
		print "	-all: to show all function"