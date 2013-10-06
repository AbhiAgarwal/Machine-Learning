import sys, time, math
from copy import deepcopy
train_vectors = [] # list of validation & training vectors
test_vectors = [] # list of test vectors
validation_vectors = [] # training vector
training_vectors = [] # validation vector

# @Gets data from train & test
# @Returns nothing
def getData():
	temp = []
	global train_vectors, test_vectors # Editing the global varabile
	print "		-> Getting Test, Validation & Training Data from MNIST"
	startTime = time.time() # Starts Timer
	train = open('./data/mnist_train.txt', 'r') # Declaration of files
	test = open('./data/mnist_test.txt', 'r') # Declaration of files
	print "		-> Starting to get Training & Validation Data"
	for i, sent in enumerate(train):
		del temp[:] # to refresh the queue
		lineOne = sent.split(",")
		temp = [0 for i in range(0, len(lineOne))]
		for x, num in enumerate(lineOne):
			if x > 0:
				vec_math = (((int(num) * 2) / (255)) - 1)
				temp[x] = vec_math
			else:
				pass
		train_vectors.append(deepcopy(temp))
	print "		-> Got Training & Validation Data"
	temp = []
	print "		-> Starting to get Test Data"
	for i, sent in enumerate(test):
		del temp[:] # to refresh the queue
		lineOne = sent.split(",")
		temp = [0 for i in range(0, len(lineOne))]
		for x, num in enumerate(lineOne):
			if x > 0:
				vec_math = (((int(num) * 2) / (255)) - 1)
				temp[x] = vec_math
			else:
				pass
		test_vectors.append((deepcopy(temp)))
	print "			-> Got Test Data"
	print "			-> There are " + str(len(train_vectors)) + " vectors for the Validation Set"
	print "			-> There are " + str(len(test_vectors)) + " vectors for the Test Set"
	print "			-> Get Data Took " + str(time.time() - startTime) + " seconds"
	return

# Splits vector training, and validation into 1600:400 split
# @Returns nothing
def createVector():
	if "-all" in sys.argv:
		print "Calling createVector"
	global validation_vectors, training_vectors
	validation_vectors = train_vectors[1:1600]
	training_vectors = train_vectors[1600:2000]
	return

# Adds 2 Vectors Together
# @ Returns added vector
def vectorAdd(vecOne, vecTwo):
	if "-all" in sys.argv:
		print "Calling vectorAdd"
	vec = []
	for one, two in zip(vecOne, vecTwo):
		vec.append(one + two)
	return vec

# Multiplies one Scalar and one Vector
# @Returns Multiplied Number
def vectorMult(scalarQ, vecAr):
	if "-all" in sys.argv:
		print "Calling vectorMult"
	for index, vector in enumerate(vecAr):
		vecAr[index] *= scalarQ
	return vecAr

# Performs dot product on two Vectors
# @Returns Dot
def dotFunction(vecOne, vecTwo):
	if "-all" in sys.argv:
		print "Calling dotFunction"
	return sum(one * two for one, two in zip(vecOne, vecTwo))

# Gets magnitude of a vector
# @Returns magnitude
def vectorMagnitude(vector):
	if "-all" in sys.argv:
		print "Calling vectorMagnitude"
	div = float(0)
	for values in vector:
		div += float(math.pow(values, 2))
	magnitude = float(math.sqrt(div))
	return 0.001 if (div == 0.0) else magnitude

# Evaluates w/ weight, lambda, and next(two)
# @Returns evaluated object
def evaluateFunc(weight, lambd, two):
	if "-all" in sys.argv:
		print "Calling evaluateFunc"
	obj = ((lambd / 2) * math.pow(vectorMagnitude(weight), 2)) + two
	return obj

# @Trains using Support Vector Machine Algorithm
# @Returns final classification vector
def pegasos_svm_train(data, lambd, iteration):
	print "		-> Starting to Train Pegasos Algorithm"
	startTime = time.time() # Starts Timer


	# FILL IN HERE NOW

	print iteration
	sys.exit(1)
	print "			-> Pegasos SVM Took " + str(time.time() - startTime) + " seconds"

# Performs & Sorts data to put into the "pegasos_svm_train" algorithm
# @Returns evaluted "pegasos_svm_train" w/ altered "multiclass-pegasos" data
def multipegasos(data, lambd, classificationNum, dataset, iterations):
	print "		-> Starting to parse data for Pegasos Algorithm"
	print "		-> Dataset: " + str(dataset)
	print "		-> Classification Number: " + str(classificationNum)
	startTime = time.time() # Starts Timer
	temp = [() for i in range(0, len(data))]
	classification = int(classificationNum) # make sure not string (error checking)
	for i, val in enumerate(data):
		temp_x = 0 #initialize
		if(int(val[0]) != classification):
			temp_x = (-1, val[1])
		else:
			temp_x = (1, val[1])
		temp[i] = deepcopy(temp_x)
	print "		-> Calling Pegasos SVM to start training"
	print "			-> Multi Took " + str(time.time() - startTime) + " seconds"
	return (pegasos_svm_train(temp, lambd, iterations))

# Tests the Pegasos SVM Algorithm to see how well the data did
# @Returns how many times it misses, but prints out the final result
def multiclass_pegasos_test(data, weights, classificationNum):
	print "heh"

# @Main function
# @Returns nothing
if __name__ == '__main__':
	if "-run" in sys.argv:
		# Initializes the program
		print "Calling: " + str(sys.argv) + " arguments."
		print "-> Starting Training & Validation Process"
		startTime = time.time() # Starts Timer
		# Start Process for user input
		getData()
		lambd = math.pow(2, -3)
		if "-lambd" in sys.argv:
			key = sys.argv.index('-lambd') + 1
			lambd = int(math.pow(2, float(sys.argv[key])))
		iteration = 5
		if "-iteration" in sys.argv:
			key = sys.argv.index('-iteration') + 1
			iteration = int(float(sys.argv[key]))
		# Runs Multi Pegasos
		# Calculate 10 weights
		weight0 = multipegasos(train_vectors, lambd, 0, "Training Data", iteration)
		weight1 = multipegasos(train_vectors, lambd, 1, "Training Data", iteration)
		weight2 = multipegasos(train_vectors, lambd, 2, "Training Data", iteration)
		weight3 = multipegasos(train_vectors, lambd, 3, "Training Data", iteration)
		weight4 = multipegasos(train_vectors, lambd, 4, "Training Data", iteration)
		weight5 = multipegasos(train_vectors, lambd, 5, "Training Data", iteration)
		weight6 = multipegasos(train_vectors, lambd, 6, "Training Data", iteration)
		weight7 = multipegasos(train_vectors, lambd, 7, "Training Data", iteration)
		weight8 = multipegasos(train_vectors, lambd, 8, "Training Data", iteration)
		weight9 = multipegasos(train_vectors, lambd, 9, "Training Data", iteration)
		# finishing off the Pegasos
		print "-> Whole Algorithm: Took " + str(time.time() - startTime) + " seconds"
	elif "-test" in sys.argv:
		print "Calling: " + str(sys.argv) + " arguments."
		print "-> Starting Test Process"
	else:
		print "	-run: to run training"
		print "	-test: to run testing"
		print "	-all: to show all function"
		print " -lambd: to input lambda (default: -3)"
		print " -iteration: to input iterations to make (default: 5)"