import torch
import torch.nn as nn
import torch.nn.functional as F


class textCNN(nn.Module):
	"""docstring for CNN"""
	def __init__(self, vocab_size,embed_size,seq_len,labels,weight,droput,**kwargs):
		super(textCNN, self).__init__(**kwargs)
		self.labels = labels
		self.embedding = nn.Embedding(vocab_size,embed_size)
		#self.embedding = nn.Embedding.from_pretrained(weight)
		self.embedding.weight.requires_grad = False
		self.conv1 = nn.Conv2d(1,1,(3,embed_size))
		self.conv2 = nn.Conv2d(1,1,(4,embed_size))
		self.conv3 = nn.Conv2d(1,1,(5,embed_size))
		self.pool1 = nn.MaxPool2d((seq_len - 3 + 1, 1))
		self.pool2 = nn.MaxPool2d((seq_len - 4 + 1, 1))
		self.pool3 = nn.MaxPool2d((seq_len - 5 + 1, 1))
		self.dropout = nn.Dropout(droput)
		self.linear = nn.Linear(3,labels)

	def forward(self,inputs):
		inputs = self.embedding(inputs).view(inputs.shape[0],1,inputs.shape[1],-1)

		x1 = F.relu(self.conv1(inputs))
		x2 = F.relu(self.conv2(inputs))
		x3 = F.relu(self.conv3(inputs))

		x1 = self.pool1(x1)
		x2 = self.pool2(x2)
		x3 = self.pool3(x3)

		x = torch.cat((x1,x2,x3),1)
		x = x.view(inputs.shape[0], 1, -1)
		x = self.dropout(x)

		x = self.linear(x)
		#logit = F.log_softmax(x, dim=1)
		x = x.view(-1, self.labels)
		
		return(x)
