
import re, nltk, random, string, time
# import numpy as np
import sys

class Profanity:


	def __init__(self,  flag):
		highlights = ['hindu', 'muslim', 'jain', 'christian', 'muslims', 'parsi']
		self.highlights = highlights
		self.near = ['near', 'close', 'nearby', 'next', 'behind', 'opposite']
		self.only = 'only'
		self.exclude = set(string.punctuation)
		self.nonPunct = re.compile('[A-Za-z0-9]+')

		self.trainClassifier(flag)


	def extractFeatures(self, sentences, label):
		featureSet = []
		for sentence in sentences:
			feature = self.extractFeature(sentence)
		# print feature
			featureS = (feature, label)		
			featureSet.append(featureS)

		return featureSet



	def predict(self, test):
		proccesedTest = self.processSent(test)
		featureSentence = self.extractFeature(proccesedTest)
		if(len(featureSentence) == 0):
			result = 'false'
		else :
			result = self.classifier.classify(featureSentence)
		return result




	def predictDescription(self, text):	
		sentences = text.rstrip().split('.')
		print sentences
		for sentence in sentences:
			result = self.predict(sentence)
			if result == 'true':
				print sentence
				return 'true'


		return 'false' 




	def extractFeature(self, sentence):
		feature = {}
		# feature = {"precede": "",
		# 		"isonly" : 0,
		# 		"isnear" : 0,
		# 		"succede": ""}

		highlights = self.highlights
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

					for word in self.impWords:
						feature["contains(%s)" % word] = (word in sentence)



					for type in self.near:
						if type in sentenceSub:
							feature['isnear'] = 1
							break

					if self.only in sentenceSub:
						feature['isonly'] = 1

					if (index >= 1):
						feature['precede'] = tokens[index-1]


					if  index < len(tokens) -1 :
						feature['succede'] = tokens[index + 1]

		return feature


	def processSentences(self, sentences):
		processedSent = []
		for sentence in sentences:
			sentence = sentence.replace('.', ' ')
			words = nltk.word_tokenize(sentence)
			filtered = [word.lower().strip('.') for word in words if self.nonPunct.match(word)]
			sentence = ' '.join(filtered)
			processedSent.append(sentence)

		return processedSent


	def processSent(self, sentence):
		proSent = sentence.replace('.', ' ')
		words = nltk.word_tokenize(proSent)
		filtered = [word.lower().strip('.') for word in words if self.nonPunct.match(word)]
		proSent = ' '.join(filtered)
		return proSent


	def trainClassifier(self, classifierType):
		f3 = open('false.txt', 'r')

		raw3 = f3.read()
		f3.close()

		tokensFalse = nltk.wordpunct_tokenize(raw3)
		fd = nltk.FreqDist(word.lower() for word in tokensFalse )
		freqWordsFalse = fd.keys()[:35]


		f4 = open('true.txt', 'r')
		raw4 = f4.read()
		f4.close()
		tokensTrue = nltk.wordpunct_tokenize(raw4)
		sentencesTrue = raw4.rstrip().split('\n')

		fd = nltk.FreqDist(word.lower() for word in tokensTrue )
		freqWordsTrue = fd.keys()[:35]

		uniqueWordsPos = [word for word in freqWordsTrue if (word not in freqWordsFalse and word not in self.highlights and len(word) > 2)]
		uniqueWordsNeg = [word for word in freqWordsFalse if (word not in freqWordsTrue and word not in self.highlights and len(word) > 2)]

		posWords = uniqueWordsPos[:8]
		negWords = uniqueWordsNeg[:5]

		impWords = posWords + negWords
		self.impWords = impWords

		sentencesFalse = raw3.rstrip().split('\n')

		processedFalseSent = []
		processedSent = []
		featureSet = []
		falseSet = []
		trueSet = []

		processedFalseSent = self.processSentences(sentencesFalse)
		falseSet = self.extractFeatures(processedFalseSent, 'false')
			
		processedTrueSent = self.processSentences(sentencesTrue)
		trueSet = self.extractFeatures(processedTrueSent,  'true')

		finalSet = falseSet + trueSet
		random.shuffle(finalSet)

		classifierList = []

		random.shuffle(finalSet)
		train_set, test_set = finalSet[:400], finalSet[400:]
		classifierList.append(nltk.NaiveBayesClassifier.train(train_set))
		fs = [ feature for (feature, label) in test_set]
		print nltk.classify.accuracy(classifierList[0], test_set)
		# NaiveBayesClassifierccuracy.append(nltk.classify.accuracy(classifier, test_set))
		s1 = classifierList[0].classify(fs[1])
		# classifierList[0].show_most_informative_features(10)

		# classifierList.append(nltk.DecisionTreeClassifier.train(train_set))
		# print nltk.classify.accuracy(classifierList[1], test_set)

		if classifierType == 1:
			classifier = classifierList[0]
		else: 
			classifier = classifierList[1]


		for (feature, label) in test_set:
			predLabel = classifier.classify(feature)
			# if predLabel != label:
		 # 		print feature, label, predLabel

		self.classifier = classifier
		self.impWords = impWords



	def demo(self):
		sentenceTest = "3 bhk flat for sale in near christian college,Calicut."
		if(len(sys.argv) >= 2):
			sentenceTest = sys.argv[1]

		self.trainClassifier()
		r2 = predictDescription(sentenceTest, impWords)
		# print r2
 
# helper functions

if __name__ == '__main__':
	flag = 1
	if len(sys.argv) > 1:
		flag = sys.argv[1]

	demoText = "Very spacious apartment in the heart of mumbai. Affordable 2 bhk. This property is only for muslims"

	# print sys.argv
	if len(sys.argv) >= 3:
		demoText = sys.argv[2]

	highlights = ['hindu', 'muslim', 'jain', 'christian', 'muslims', 'parsi']
	profane = Profanity(flag)
	demoResult = profane.predictDescription(demoText)
	print demoResult
				





