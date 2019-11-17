import tensorflow as tf
from model import InferSent
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
if __name__ == '__main__':
	config = {}
	config['maxlen'] = 10
	config['hidden_size'] = 100
	config['lr'] = 1e-3
	config['vocab_size'] = 1000
	config['dropout'] = 0.8
	config['kernel'] = 3
	config['class_size'] = 3
	config['batch_size'] = 64

	my_model = InferSent(config)

	my_model.init_placeholder()

	my_model.build_graph()