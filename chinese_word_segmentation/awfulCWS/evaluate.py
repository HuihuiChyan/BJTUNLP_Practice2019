
def cac_gold_num(tags):
	gold_num = 0
	for tag in tags:
		if int(tag) == 3:
			gold_num += 1
		if int(tag) == 0:
			gold_num += 1
	return gold_num

def cac_pre_num(tags):
	pre_num = 0
	flag = 0
	for tag in tags:
		if int(tag) == 3:
			pre_num += 1
		if (int(tag) == 3 or int(tag)==0) and flag == 1:
			flag = 0
		if int(tag) == 0:
			flag = 1
		if int(tag) == 1:
			flag = 1
			pre_num += 1
			flag = 0
	return pre_num

def tag2sent(feature,tag,length,vocab):
	#word2idx = {val:key for key,val in vocab}
	sent = feature[:length]
	idx2sent = []
	k = 0
	flag = 0
	for ch in tag:
		if ch == 3:
			idx2sent.append(vocab[feature[k]])
		elif ch == 0:
			flag = 1
			st = k
		elif ch==1 and flag==1:
			temp = ""
			for i in range(st,k+1):
				temp = temp + vocab[feature[i]]
			idx2sent.append(temp)
			flag = 0
			st = -1
		if flag==1 and(ch==3 or ch==1):
			flag = 0
		k += 1
	return idx2sent

def eva(features,pre_tags,tags,lengths,vocab):
	gold_num = 0
	pre_num = 0
	cor_num = 0
	#pre_tags = pre_tags.numpy().tolist()
	tags = tags.numpy().tolist()
	features = features.numpy().tolist()
	for feature,pre_tag , tag, length in zip(features,pre_tags,tags,lengths):
		cor_sent = tag2sent(feature,tag,length,vocab)
		pre_sent = tag2sent(feature,pre_tag,length,vocab)
		print(cor_sent)
		print(pre_sent)
		gold_num += len(cor_sent)
		pre_num += len(pre_sent)
		for ch in cor_sent:
			if ch in pre_sent:
				cor_num += 1
	precision = cor_num / pre_num 
	recall = cor_num / gold_num
	f1 = (2*precision*recall) / (precision + recall)
	return precision,recall,f1








def evaluation(features,pre_tags,tags,lengths):
	gold_num = 0
	pre_num = 0
	cor_num = 0
	#pre_tags = pre_tags.numpy().tolist()
	tags = tags.numpy().tolist()
	for pre_tag , tag, length in zip(pre_tags,tags,lengths):
		flag = 0
		gold_num += cac_gold_num(tag[:length])
		pre_num += cac_pre_num(pre_tag[:length])
		for pre_ch,gold_ch in zip(pre_tag,tag):
			if gold_ch == 3:
				if pre_ch == 3:
					cor_num += 1
				continue
			if gold_ch == 0:
				flag = 1
				if pre_ch!= gold_ch:
					flag = 0
					continue
			if gold_ch == 1:
				if pre_ch != gold_ch :
					flag = 0
					continue
			if gold_ch == 1 and flag == 1:
				if pre_ch == gold_ch :
					cor_num += 1
				flag = 0
	precision = cor_num / pre_num 
	recall = cor_num / gold_num
	print(cor_num)
	print(pre_num)
	print(gold_num)
	f1 = (2*precision*recall) / (precision + recall) if (precision + recall) else 0
	return precision,recall,f1
				










