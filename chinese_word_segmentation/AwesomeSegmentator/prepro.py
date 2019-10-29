import os
import pdb
import re
from collections import Counter

def preprocess():
	if not os.path.exists("./data"):
		os.mkdir("./data")
	if not os.path.exists("./data/train.txt"):
		with open("./training/msr_training.utf8","r",encoding="utf-8") as ftrain,\
		open("./data/train.txt","w",encoding="utf-8") as ftrainout,\
		open("./data/vocab.txt","w",encoding="utf-8") as fvocab:
			trainlines = [line.strip() for line in ftrain.readlines()]
			taglines = []
			newtrainlines = []
			for line in trainlines:
				if re.match(r"[\s]*$", line):
					# pdb.set_trace()
					continue
				line = re.split(r'[\s]+',line)
				tagline = []
				for word in line:
					if len(word) == 1:
						tagline.append("S")
					elif len(word) == 2:
						tagline.append("B")
						tagline.append("E")
					else:
						tagline.append("B")
						for _ in range(len(word)-2):
							tagline.append("M")
						tagline.append("E")
				taglines.append("".join(tagline))
				newtrainlines.append("".join(line))
			trainlines = newtrainlines
			for line in zip(trainlines, taglines):
				ftrainout.write(line[0]+" ||| "+line[1]+"\n")
			trainlines = [list(line) for line in trainlines]
			counterlines = []
			for line in trainlines:
				counterlines.extend(line)
			common_chars = list(Counter(counterlines).most_common())
			# pdb.set_trace()
			common_chars = common_chars[:4000]
			fvocab.write("UNK"+"\n")
			fvocab.write("PAD"+"\n")
			for char in common_chars:
				fvocab.write(char[0]+"\n")
	if not os.path.exists("./data/test.txt"):
		with open("./gold/msr_test_gold.utf8","r",encoding="utf-8") as ftest,\
		open("./data/test.txt","w",encoding="utf-8") as ftestout:
			testlines = [re.split(r'[\s]+',line.strip())for line in ftest.readlines()]
			taglines = []
			for line in testlines:
				tagline = []
				for word in line:
					if len(word) == 1:
						tagline.append("S")
					elif len(word) == 2:
						tagline.append("B")
						tagline.append("E")
					else:
						tagline.append("B")
						for _ in range(len(word)-2):
							tagline.append("M")
						tagline.append("E")
				taglines.append("".join(tagline))
			testlines = ["".join(line) for line in testlines]
			for line in zip(testlines, taglines):
				try:
					assert len(line[0]) == len(line[1])
				except:
					pdb.set_trace()
				ftestout.write(line[0]+" ||| "+line[1]+"\n")
if __name__ == "__main__":
	preprocess()