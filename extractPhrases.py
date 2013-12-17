
filename = 'newah'
f = open(filename)
raw = f.read()
f.close()

import re, nltk, random, string, time
import numpy as np

highlights = ['hindu', 'muslim', 'jain', 'christian', 'muslims']
tokens = nltk.wordpunct_tokenize(raw)
words = [w.lower() for w in tokens]
keySentences = []
sentences = nltk.sent_tokenize(raw)

for sentence in sentences:
	for highlight in highlights:
		if re.search(highlight, sentence, re.IGNORECASE):
			keySentences.append(sentence)
			break


exclude = set(string.punctuation)

f2 = open('profanitytext8.txt', 'w')
# for sentence in keySentences:
# 	f2.write(sentence)
print("raw file read")

for sentence in keySentences:
	s = ''.join(ch for ch in sentence if ch not in exclude)
	for word in nltk.word_tokenize(s):
		for highlight in highlights:
			if highlight in word.lower():
				s = ''.join(ch for ch in sentence if ch not in exclude)
				tokens = nltk.word_tokenize(s)
				index = tokens.index(word)
				exclude = set(string.punctuation)
				if index > 5:
					sentenceSub =' '.join(tokens[index-5:index+4])
				else:
					sentenceSub = ' '.join(tokens)  

				
				features = {"precede": "",
							"succede": "",
							"sentence": sentence,
							"word": tokens[index]
							}

				if (index > 1):
					features['precede'] = tokens[index-1]

				if  index < len(tokens) -1 :
					features['succede'] = tokens[index + 1]

				f2.write(features['sentence'])
				f2.write('\n')

f2.close() 

print "==========extraction complete=========="