import re, nltk, random, string, time
import numpy as np
import sys



def extractFeatures(sentences, highlights, impWords, label):
	featureSet = []
	for sentence in sentences:
		feature = extractFeature(sentence, highlights, impWords)
		# print feature
		featureS = (feature, label)		
		featureSet.append(featureS)

	# print len(featureSet)
	return featureSet


def predict(classifier, test, impWords):
	proccesedTest = processSent(test)
	featureSentence = extractFeature(proccesedTest, highlights, impWords)
	if(len(featureSentence) == 0):
		result = 'false'
	else :
		result = classifier.classify(featureSentence)
	return result




def predictDescription(classifier, text, impWords):
	
	sentences = text.rstrip().split('.')
	print sentences
	for sentence in sentences:
		result = predict(classifier, sentence, impWords)
		if result == 'true':
			print sentence
			return 'true'


	return 'false' 




def extractFeature(sentence, highlights, impWords):
	feature = {}
	# feature = {"precede": "",
	# 		"isonly" : 0,
	# 		"isnear" : 0,
	# 		"succede": ""}
	for word in nltk.word_tokenize(sentence):
		for highlight in highlights:
			if highlight in word.lower():
				tokens = nltk.word_tokenize(sentence)
				word = word.lower()
				tokens = [tk.lower() for tk in tokens]
				index = tokens.index(word)
				complete = 0
				if highlight == word.lower():
					complete =1

				if len(tokens) > 8 & index > 4:
					sentenceSub = ' '.join(tokens[index-4:index+4])
				else:
					sentenceSub = ' '.join(tokens) 	


				# tagpairs = nltk.pos_tag(tokens)

				for word in impWords:
					feature["contains(%s)" % word] = (word in sentence)



				for type in near:
					if type in sentenceSub:
						feature['isnear'] = 1
						break

				if only in sentenceSub:
					feature['isonly'] = 1

				if (index >= 1):
					feature['precede'] = tokens[index-1]


				if  index < len(tokens) -1 :
					feature['succede'] = tokens[index + 1]

	return feature


def processSentences(sentences):
	processedSent = []
	for sentence in sentences:
		sentence = sentence.replace('.', ' ')
		words = nltk.word_tokenize(sentence)
		filtered = [word.lower().strip('.') for word in words if nonPunct.match(word)]
		sentence = ' '.join(filtered)
		processedSent.append(sentence)

	return processedSent


def processSent(sentence):
	proSent = sentence.replace('.', ' ')
	words = nltk.word_tokenize(proSent)
	filtered = [word.lower().strip('.') for word in words if nonPunct.match(word)]
	proSent = ' '.join(filtered)
	return proSent



highlights = ['hindu', 'muslim', 'jain', 'christian', 'muslims', 'parsi']

exclude = set(string.punctuation)
nonPunct = re.compile('[A-Za-z0-9]+')
near = ['near', 'close', 'nearby', 'next', 'behind', 'opposite']
only = 'only'



def trainClassifier():
	f3 = open('/home/akashdhaka/R/phrase-intent-score/false.txt', 'r')

	raw3 = f3.read()
	f3.close()

	tokensFalse = nltk.wordpunct_tokenize(raw3)
	fd = nltk.FreqDist(word.lower() for word in tokensFalse )
	freqWordsFalse = fd.keys()[:35]


	f4 = open('/home/akashdhaka/R/phrase-intent-score/true.txt', 'r')
	raw4 = f4.read()
	f4.close()
	tokensTrue = nltk.wordpunct_tokenize(raw4)
	sentencesTrue = raw4.rstrip().split('\n')

	fd = nltk.FreqDist(word.lower() for word in tokensTrue )
	freqWordsTrue = fd.keys()[:35]

	uniqueWordsPos = [word for word in freqWordsTrue if (word not in freqWordsFalse and word not in highlights and len(word) > 2)]
	uniqueWordsNeg = [word for word in freqWordsFalse if (word not in freqWordsTrue and word not in highlights and len(word) > 2)]

	posWords = uniqueWordsPos[:8]
	negWords = uniqueWordsNeg[:5]

	impWords = posWords + negWords

	sentencesFalse = raw3.rstrip().split('\n')

	processedFalseSent = []
	processedSent = []
	featureSet = []
	falseSet = []
	trueSet = []

	processedFalseSent = processSentences(sentencesFalse)
	falseSet = extractFeatures(processedFalseSent, highlights, impWords, 'false')
			
	processedTrueSent = processSentences(sentencesTrue)
	trueSet = extractFeatures(processedTrueSent, highlights, impWords, 'true')

	finalSet = falseSet + trueSet
	random.shuffle(finalSet)

	BcAccuracy = []

	random.shuffle(finalSet)
	train_set, test_set = finalSet[:400], finalSet[400:]
	classifier = nltk.NaiveBayesClassifier.train(train_set)
	# print nltk.classify.accuracy(classifier, test_set)
	fs = [ feature for (feature, label) in test_set]
	print nltk.classify.accuracy(classifier, test_set)
	# NaiveBayesClassifierccuracy.append(nltk.classify.accuracy(classifier, test_set))
	s1 = classifier.classify(fs[1])
	classifier.show_most_informative_features(10)


	for (feature, label) in test_set:
		predLabel = classifier.classify(feature)
		if predLabel != label:
		 	print feature, label, predLabel


	return (classifier, impWords)



# DtAccuracy = []
# for i in range(1, 2):
# 	random.shuffle(finalSet)
# 	train_set, test_set = finalSet[:550], finalSet[550:]
# 	classifier = nltk.DecisionTreeClassifier.train(train_set)
# 	# print nltk.classify.accuracy(classifier, test_set)
# 	DtAccuracy.append(nltk.classify.accuracy(classifier, test_set))


sentenceTest = "3 bhk flat for sale in near christian college,Calicut."
if(len(sys.argv) >= 2):
	sentenceTest = sys.argv[1]

(classifier, impWords) = trainClassifier()
# r1 = predict(classifier, sentenceTest, impWords)
# print r1

r2 = predictDescription(classifier, sentenceTest, impWords)
print r2
 

# print  np.mean(DtAccuracy) 
# print  np.median(DtAccuracy)

# helper functions
#this is a great society to live, with designer kitchen , 24 hrs supply. It is available for rent. Price
#is Rs13000. Property built by xyz constructions.



				





