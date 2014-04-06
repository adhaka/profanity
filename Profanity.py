
import re, nltk, random, string, time
# import numpy as np
import json
import sys
# import memcache
# from datetime import datetime


class Profanity:

	def __init__(self,  flag):
		highlights = ['hindu', 'muslim', 'jain', 'christian', 'parsi', 'brahmin', 'sindhi', 'catholic', 'islamist', 'muhammedan', 'buddhist', 'gujrati', 'marwari', 'marwaris', 'marwadi', 'gujarati', 'Gujrathi', 'marathi', 'bengali', 'maharashtrian', 'manipuri', 'african']

		self.highlights = highlights
		self.highlightsPlu = [word + 's' for word in highlights]
		self.keyWord = 'None'
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
				tokens = nltk.word_tokenize(sentence)
				tokenSmall = [word.lower() for word in tokens]
				keyIndex = tokenSmall.index(self.keyWord)
				outText = sentence
				if len(tokens) > 9:
					if keyIndex > 2:
						outText = ' '.join(tokens[keyIndex - 2: keyIndex +5])
					else:
						outText = ' '.join(tokens[keyIndex : keyIndex +5])

				jsonResult = json.dumps(dict(result='true', text=outText))
				return jsonResult


		jsonResult = json.dumps(dict(result='false'))
		return jsonResult 

	def bayesianclassifier(self,sentence):
		words = nltk.word_tokenize(sentence)
		words = [word.lower() for word in words]
		pscoretrue ={}
		pscorefalse = {}
		for word in words:
			pscoretrue[word]= self.calculateProb(word)
			pscorefalse[word] = 1.0 - pscoretrue[word]
		# print pscore
		totalpscore = reduce(lambda x,y: x*y , pscoretrue.values()) ** (1.0/len(pscoretrue))
		totalpscoretrue = reduce(lambda x,y: x*y, pscoretrue.values()) 
		totalpscorefalse = reduce(lambda x,y: x*y, pscorefalse.values())
		sentscore = float(totalpscoretrue)/(totalpscorefalse + totalpscoretrue)
		print totalpscore
		print sentscore

		return totalpscore

	def calculateProb(self, word):
		# print self.trainingdata['true']
		truecount = float(len([True for sentence in self.trainingdata['true'] if word in sentence]))
		falsecount = float(len([True for sentence in self.trainingdata['false'] if word in sentence]))
		if truecount == 0:
			truecount = 1 

		if falsecount == 0:
			falsecount = 1

		truefreq = float(truecount)/len(self.trainingdata['true'])
		falsefreq = float(falsecount)/len(self.trainingdata['false'])
		print truecount, falsecount, truefreq, falsefreq
	#float trueFreq = self.tokensTrue.count(word)/len(self.tokensTrue)
	#float falseFreq = self.tokensFalse.count(word)/len(self.tokensFalse)
		score = truefreq/(truefreq + falsefreq)


		return score

	def extractFeature(self, sentence):
		feature = {}
		highlights = self.highlights
		for word in nltk.word_tokenize(sentence):
			for highlight in highlights:
				if highlight in word.lower():
					tokens = nltk.word_tokenize(sentence)
					self.keyWord = word
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
						
					if (index >= 2):
						feature['precede2'] = tokens[index - 2]

					if  index < len(tokens) -1 :
						feature['succede'] = tokens[index + 1]
	        return feature



	def processSentences(self, sentences):
		processedSent = []
		for sentence in sentences:
			sentence = sentence.replace('.', ' ')
			sentence = sentence.replace('to', ' ')
			# sentence = sentence.replace('is', ' ')
			sentence = sentence.replace('are', ' ')
			words = nltk.word_tokenize(sentence)
			filtered = [word.lower().strip('.') for word in words if self.nonPunct.match(word)]
			sentence = ' '.join(filtered)
			processedSent.append(sentence)

		return processedSent



	def processSent(self, sentence):
		proSent = sentence.replace('.', ' ')
		proSent = proSent.replace('to', ' ')
		# proSent = proSent.replace('is', ' ')
		proSent = proSent.replace('are', ' ')
		words = nltk.word_tokenize(proSent)
		filtered = [word.lower().strip('.') for word in words if self.nonPunct.match(word)]
		proSent = ' '.join(filtered)
		return proSent


	def trainClassifier(self, classifierType):
		
		f3 = open('false.txt', 'r')
		rawFalse = f3.read()
		f3.close()

		self.tokensFalse = nltk.wordpunct_tokenize(rawFalse)

		fd = nltk.FreqDist(word.lower() for word in self.tokensFalse )
		freqWordsFalse = fd.keys()[:35]

		f4 = open('true.txt', 'r')
		rawTrue = f4.read()
		f4.close()

		self.tokensTrue = nltk.wordpunct_tokenize(rawTrue)

		sentencesTrue = rawTrue.rstrip().split('\n')

		fd = nltk.FreqDist(word.lower() for word in self.tokensTrue )
		freqWordsTrue = fd.keys()[:35]

		uniqueWordsPos = [word for word in freqWordsTrue if (word not in freqWordsFalse and word not in self.highlights and word not in self.highlightsPlu and len(word) > 2)]
		uniqueWordsNeg = [word for word in freqWordsFalse if (word not in freqWordsTrue and word not in self.highlights and word not in self.highlightsPlu  and len(word) > 2)]

		posWords = uniqueWordsPos[:10]
		negWords = uniqueWordsNeg[:7]

		impWords = posWords + negWords
		self.impWords = impWords

		sentencesFalse = rawFalse.rstrip().split('\n')

		processedFalseSent = []
		processedSent = []
		featureSet = []
		falseSet = []
		trueSet = []

		self.trainingdata ={}
		processedFalseSent = self.processSentences(sentencesFalse)
		falseSet = self.extractFeatures(processedFalseSent, 'false')
		processedTrueSent = self.processSentences(sentencesTrue)
		trueSet = self.extractFeatures(processedTrueSent,  'true')

		finalSet = falseSet + trueSet
		random.shuffle(finalSet)

		self.trainingdata['true'] = processedTrueSent
		self.trainingdata['false'] = processedFalseSent

		classifierList = []

		random.shuffle(finalSet)
		train_set, test_set = finalSet[:750], finalSet[750:]
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
		r2 = self.predictDescription(sentenceTest)
		# print r2

# helper functions

if __name__ == '__main__':
	flag = 1
	if len(sys.argv) > 2:
		flag = sys.argv[2]

	demoText = "Very spacious apartment in the heart of mumbai. Affordable 2 bhk. This property is only for muslims"

	# print sys.argv
	if len(sys.argv) >= 2:
		demoText = sys.argv[1]

	highlights = ['hindu', 'muslim', 'muslims', 'jain', 'christian', 'parsi', 'brahmin', 'sindhi', 'catholic', 'islamist', 'muhammedan', 'buddhist', 'gujrati', 'marwari', 'marwadi', 'gujarati', 'Gujrathi', 'marathi', 'bengali', 'maharashtrian', 'manipuri', 'african']
	profane = Profanity(flag)
	demoResult = profane.predictDescription(demoText)
	profane.bayesianclassifier(demoText)
	print profane.extractFeature(demoText)
	print demoResult
				






