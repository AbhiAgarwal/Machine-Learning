# Checks if featureWord function is correct
# and checks
def checkAndSet(number):
	word_set = (training_data[number].split())
	check = bool(1)
	count = 1
	while check == bool(1):
		if word_set.__len__() > count:
			feature_set = feature_list[number]
			for c, element in enumerate(feature_set):
				if feature_set[c] in vocabulary_list:
					count += 1
				elif word_set[count] not in vocabulary_list:
					print word_set[count]
					check = bool(0)
					print "Fake"
					return 0
		else:
			print check