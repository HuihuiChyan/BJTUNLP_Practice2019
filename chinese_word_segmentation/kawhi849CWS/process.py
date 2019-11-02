
def build_vocab():
	w = open('../vocab.txt','w')
	vocab = set()
	for line in open('train.txt'):
		string = ''.join(line.strip().split(' '))
		for i in string:
			if i in vocab:
				continue
			else:
				w.write(i+'\n')
				vocab.add(i)

build_vocab()



