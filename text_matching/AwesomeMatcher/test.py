import prepro
def test_dict():
	path = './vocab_all.txt'
	vocab = prepro.read_dict(path)
	for i in range(len(vocab)):
		print(vocab[i])

def test_read():
	path = './data/lcmqc/test.txt'
	res = prepro.get_train_set(path)
	for i in range(len(res)):
		print(res[i].text_a, '\t', res[i].text_b, '\t', res[i].label)

def test_sent2idx():
	vocab_path = './vocab_all.txt'
	idx2word_vocab = prepro.read_dict(vocab_path)

	word2idx_vocab = {value:key for key,value in idx2word_vocab.items()}

	data_path = './data/lcmqc/dev.txt'
	train_set = prepro.get_train_set(data_path)

	sent_a_idxs, sent_b_idxs, label = prepro.turn2idx(train_set, word2idx_vocab, 10)

	for i in range(3):
		print("word:", train_set[i].text_a)
		print("idx:",sent_a_idxs[i])
		print("word",train_set[i].text_b)
		print("idx:",sent_b_idxs[i])
		print(label[i])

	for sent_a_batch, sent_b_batch, label_batch in prepro.generate_batch(sent_a_idxs, sent_b_idxs, label, batch_size=4):
		print(sent_a_batch)
		print(label_batch)
		exit()
test_sent2idx()