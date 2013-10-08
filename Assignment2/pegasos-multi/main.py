import sys, time, math, operator
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
		toAppend = [0 for i in range(0, 2)]
		toAppend[0] = int(lineOne[0]) # takes '3', '[-1, 1, 1, -1, 0]' to Append
		toAppend[1] = deepcopy(temp)
		train_vectors.append(toAppend)
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
		toAppend = [0 for i in range(0, 2)]
		toAppend[0] = int(lineOne[0]) # takes '3', '[-1, 1, 1, -1, 0]' to Append
		toAppend[1] = deepcopy(temp)
		test_vectors.append(toAppend)
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
	base_0 = data[0]
	base_1 = base_0[1]
	iterations = 1
	weights_pegasos = [0 for i in range(0, len(base_1))]
	u = [0 for i in range(0, len(base_1))]
	t = 0
	nt = 0
	for i in range(0, iteration): # 20 passes through data
		overone = 0 # > 1
		underone = 0 # < 1
		evalution = 0 # eval
		for index, vector in enumerate(data): # Traversal through the Feature List
			yj = vector[0]
			vec = vector[1]
			t += 1
			nt = ((float(1))/(t * lambd))
			if((yj * dotFunction(weights_pegasos, vec)) < 1):
				u = vectorAdd((vectorMult((1 - (nt * lambd)), weights_pegasos)), (vectorMult((nt * yj), vec)))
				underone += 1
			else:
				u = vectorMult((1 - (nt * lambd)), (weights_pegasos))
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
def multipegasostest(data, weights, classifications, dataset):
	print "		-> Starting to parse data for Pegasos Tester Algorithm"
	print "		-> Dataset: " + str(dataset)
	startTime = time.time() # Starts Timer
	errors = 0
	hand = []
	for i, val in enumerate(data):
		print val
		variable_x = val[0] # Placeholder for x when x is [x, [0,1,-1,1,0...]]
		vector_x = val[1] # Vector [variable_x, [vector_x]]
		hand = [0 for i in range(0, len(weights))] # initialize hand
		for x, weightOne in enumerate(weights):
			hand[x] = dotFunction(weightOne, vector_x)
		max_x, max_y = max((enumerate(hand)), key = operator.itemgetter(1))
		if((int(classifications[max_x])) != (int(variable_x))):
			errors += 1
	notRight = (float(errors) / (float(len(data))))
	print "			-> Error: " + str(errors) + ", out of: " + str(len(data)) + ". Has " + str(notRight) + "% errors"
	print "			-> Took " + str(time.time() - startTime) + " seconds"	
	return errors

# @Main function
# @Returns nothing
if __name__ == '__main__':
	if "-run" in sys.argv:
		# Initializes the program
		print "Calling: " + str(sys.argv) + " arguments."
		print "-> Starting Training & Validation Process"
		startTime = time.time() # Starts Timer
		# Gets data of mnist_test & mnist_train
		getData()
		# Start Process for user input
		# Get Lambda
		lambd = math.pow(2, -3)
		if "-lambd" in sys.argv:
			key = sys.argv.index('-lambd') + 1
			lambd = int(math.pow(2, float(sys.argv[key])))
		# Get the classification index you want
		classification_index = []
		classification_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
		if "-classification" in sys.argv:
			key = sys.argv.index('-classification') + 1
			classification_index = list(sys.argv[key])
			classification_index = map(int, classification_index)
			if (len(classification_index) != 10):
				classification_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
				print "		-> *** Classification Size Error: set to default"
		# Get number of Iterations
		iteration = 5
		if "-iteration" in sys.argv:
			key = sys.argv.index('-iteration') + 1
			iteration = int(float(sys.argv[key]))
		# Runs Multi Pegasos
		# Calculate 10 weights
		weight0 = multipegasos(train_vectors, lambd, classification_index[0], "Training Data", iteration)
		weight1 = multipegasos(train_vectors, lambd, classification_index[1], "Training Data", iteration)
		weight2 = multipegasos(train_vectors, lambd, classification_index[2], "Training Data", iteration)
		weight3 = multipegasos(train_vectors, lambd, classification_index[3], "Training Data", iteration)
		weight4 = multipegasos(train_vectors, lambd, classification_index[4], "Training Data", iteration)
		weight5 = multipegasos(train_vectors, lambd, classification_index[5], "Training Data", iteration)
		weight6 = multipegasos(train_vectors, lambd, classification_index[6], "Training Data", iteration)
		weight7 = multipegasos(train_vectors, lambd, classification_index[7], "Training Data", iteration)
		weight8 = multipegasos(train_vectors, lambd, classification_index[8], "Training Data", iteration)
		weight9 = multipegasos(train_vectors, lambd, classification_index[9], "Training Data", iteration)
		# Form all Weights
		# into Array, and sets the classification
		allWeights = (weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9)
		errors = multipegasostest(test_vectors, allWeights, classification_index, "Test Data")
		# finishing off the Pegasos
		print "-> Whole Algorithm: Took " + str(time.time() - startTime) + " seconds"
	elif "-test" in sys.argv:
		print "Calling: " + str(sys.argv) + " arguments."
		print "-> Starting Test Process"
	else:
		print "	-run: to run training"
		print "	-test: to run testing"
		print "	-all: to show all function"
		print "	-lambd: to input lambda (default: -3)"
		print "	-iteration: to input iterations to make (default: 5)"
		print "	-classification: to enter classifications, enter as 0123456789 (joint), (default: 0123456789)"