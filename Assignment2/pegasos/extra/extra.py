# @Create feature set of email set (x with sub(i))
# @Returns nothing
def oldfeatureWord(email_list, dataset):
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