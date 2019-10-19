
def read_file(data_path, label_path):
	texts = []
	labels = []
	for line in open(data_path):
		texts.append(line.strip().split(' '))
	for line in open(label_path):
		labels.append(line.strip().split(' '))
	return texts, labels

def read_vocab(data_path, label_path):
	vocab = dict()
	vocab_t = dict()
	num = 0
	for line in open(data_path):
		vocab[line.strip()] = num
		num += 1
	num = 0
	for line in open(label_path):
		vocab_t[line.strip()] = num
		num += 1
	return vocab, vocab_t

def sent2vec(sents, labels, vocab, vocab_t, max_len):
	vecs = []
	lens = []
	for sent in sents:
		vec = [vocab[word] if word in vocab else 0 for word in sent]
		length = len(vec)
		if length <= max_len:
			vec += [0 for _ in range(max_len - length)]
		else:
			vec = vec[:max_len]
		vecs.append(vec)
		lens.append(length)
	tags = []
	for label in labels:
		tag = [vocab_t[l] for l in label]
		if len(tag) <= max_len:
			tag += [0 for _ in range(max_len - len(tag))]
		else:
			tag = tag[:max_len]
		tags.append(tag)
	return vecs, lens, tags

def generator(data, batch_size=64):
	seqs, seq_lens, labels = [], [], []
	for sent, seq_len, label in data:
		if len(seqs) == batch_size:
			yield seqs, seq_lens, labels
			seqs, seq_lens, labels = [], [], []
		seqs.append(sent)
		seq_lens.append(seq_len)
		labels.append(label)
	if len(seqs) != 0:
		yield seqs, seq_lens, labels


