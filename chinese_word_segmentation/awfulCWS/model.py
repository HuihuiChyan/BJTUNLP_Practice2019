import torch
import torch.nn as nn
START_TAG = "<START>"
STOP_TAG = "<STOP>"
#tag_to_idx = {"B": 0, "E": 1, "M": 2, "S": 3,START_TAG: 4, STOP_TAG: 5}



def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

class Bilstm(nn.Module):
	"""docstring for Bilstm"""
	def __init__(self, vocab_size,maxlen,hidden_size,embedding_dim,num_layers,class_nums,tag_to_idx,batch_size,device):
		super(Bilstm, self).__init__()
		self.vocab_size = vocab_size
		self.maxlen = maxlen
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.class_nums = class_nums
		self.embedding_dim = embedding_dim
		self.tag_to_idx = tag_to_idx
		self.device = device
		#self.batch_size = batch_size

		self.embedding = nn.Embedding(vocab_size,embedding_dim)
		self.lstm = nn.LSTM(embedding_dim,hidden_size//2,num_layers,bidirectional=True,batch_first=True)
		self.linear = nn.Linear(hidden_size,class_nums)
		self.transitions = nn.Parameter(torch.randn(class_nums,class_nums,device=self.device))
		#self.transitions.data[tag_to_idx[START_TAG], :] = -10000
		#self.transitions.data[:, tag_to_idx[STOP_TAG],] = -10000

		self.transitions.data[self.tag_to_idx[START_TAG], :] = -10000
		self.transitions.data[:, self.tag_to_idx[STOP_TAG]] = -10000


	def init_hidden(self):
		return(torch.randn(2*self.num_layers,self.batch_size,self.hidden_size//2,device=self.device),torch.randn(2*self.num_layers,self.batch_size,self.hidden_size//2,device=self.device))

	def set_batch_size(self,sentence):
		tmp = sentence.size()
		self.seq_length = tmp[1]
		self.batch_size = tmp[0]


	def _forward_alg(self,feats):
		init_alphas = torch.full((1,self.class_nums),-10000.,device=self.device)
		init_alphas[0][self.tag_to_idx[START_TAG]] = 0
		forward_var = init_alphas
		for feat in feats:
			alphas_t = []
			for next_tag in range(self.class_nums):
				emit_score = feat[next_tag].view(1,-1).expand(1,self.class_nums)
				trans_score = self.transitions[next_tag].view(1,-1)
				next_tag_var = forward_var + trans_score + emit_score
				alphas_t.append(log_sum_exp(next_tag_var).view(1))
			forward_var = torch.cat(alphas_t).view(1,-1)
		terminal_val = forward_var + self.transitions[self.tag_to_idx[STOP_TAG]]
		alpha = log_sum_exp(terminal_val)
		return alpha

	#得到bi-lstm提取特征
	def _get_lstm_features(self,sentence):
		self.hidden = self.init_hidden()
		#[batch_size,maxlen]
		#[batch_size,maxlen,embedding_size]
		embeds = self.embedding(sentence)
		#.view(-1,self.maxlen,self.embedding_dim)
		#embeds = self.embedding(sentence).view(1,len(sentence),-1)

		lstm_out,self.hidden = self.lstm(embeds,self.hidden)

		#[batch_size,maxlen,hidden_size]
		#lstm_out = lstm_out.view(-1,self.maxlen,self.hidden_size)
		lstm_feats = self.linear(lstm_out)
		return lstm_feats

	#计算真实路径得分
	def _score_sentence(self,feats,tags):
		score = torch.zeros(1,device=self.device)
		#拼接函数
		tags = torch.cat([torch.tensor([self.tag_to_idx[START_TAG]],dtype=torch.long,device=self.device),tags])

		for i,feat in enumerate(feats):
			score = score + self.transitions[tags[i+1],tags[i]] + feat[tags[i+1]]
			
		score = score + self.transitions[self.tag_to_idx[STOP_TAG], tags[-1]]
		
		return score

	def neg_log_likelihood(self,sentence,tags):
		self.set_batch_size(sentence)
		feats = self._get_lstm_features(sentence)
		#print(feats.size())
		
		tags = tags.view(1,-1)
		#print(tags.size())
		forward_scores = []
		gold_scores = []
		score = 0
		n = 0
		for feat,tag in zip(feats,tags):
			n += 1
			forward_score = self._forward_alg(feat)
			gold_score = self._score_sentence(feat,tag)
			temp = forward_score - gold_score
			score += temp	
		return score/n

	def _viterbi_decode(self,feats):
		# feats [maxlen,class_nums]
		backpointers = []
		init_vvars = torch.full((1,self.class_nums),-10000,device=self.device)
		init_vvars[0][self.tag_to_idx[START_TAG]] = 0

		forward_var = init_vvars

		for feat in feats:
			bptrs_t = []
			viterbivars_t = []

			for next_tag in range(self.class_nums):
				next_tag_var = forward_var + self.transitions[next_tag]
				best_tag_id = argmax(next_tag_var)
				bptrs_t.append(best_tag_id)
				viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
			forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
			backpointers.append(bptrs_t)

		terminal_var = forward_var + self.transitions[self.tag_to_idx[STOP_TAG]]
		best_tag_id = argmax(terminal_var)
		path_score = terminal_var[0][best_tag_id]

		best_path = [best_tag_id]

		for bptrs_t in reversed(backpointers):
			best_tag_id = bptrs_t[best_tag_id]
			best_path.append(best_tag_id)
        
		start = best_path.pop()
		assert start == self.tag_to_idx[START_TAG] 
		best_path.reverse()
		return path_score, best_path

	def forward(self,sentence):
		self.set_batch_size(sentence)
		lstm_feats = self._get_lstm_features(sentence)
		scores = []
		tag_seqs = []
		for lstm_feat in lstm_feats:
			score, tag_seq = self._viterbi_decode(lstm_feat)
			scores.append(score)
			tag_seqs.append(tag_seq)
		return scores, tag_seqs








		