from util import *

class Dataset:

	def __init__(self, 
		fld_data, 
		max_seq_len=[150,32],
		):

		self.path_vocab = os.path.join(fld_data, 'vocab.txt')
		self.path_train = os.path.join(fld_data, 'train.num')
		self.path_test = os.path.join(fld_data, 'test.num')

		# load token dictionary

		self.max_seq_len = max_seq_len
		self.index2token, self.token2index = load_vocab(self.path_vocab)

		self.SOS = self.token2index[SOS_token]
		self.EOS = self.token2index[EOS_token]
		self.num_tokens = len(self.token2index)	# not including 0-th

		# load source-target pairs, tokenized
		
		self.reset('train')
		self.reset('test')


	def reset(self, task):
		self.generator = {
			'train': line_generator(self.path_train),
			'test': line_generator(self.path_test),
			}


	def seq2text(self, seq):
		words = []
		for j in seq:
			if j > 0:
				words.append(self.index2token[int(j)])
		return ' '.join(words)

	
	def text2seq(self, text):
		tokens = text.strip().split(' ')
		seq = []
		ix_unk = self.token2index[UNK_token]
		for token in tokens:
			seq.append(self.token2index.get(token, ix_unk))
		return seq


	def load_data(self, task, max_n):

		data_src = np.zeros((max_n, self.max_seq_len[0]))
		data_tgt = np.zeros((max_n, self.max_seq_len[1]))
		labels = []
		i = 0
		for line in self.generator[task]:
			src, tgt, label = line.strip('\n').split('\t')
			words = src.split()
			n_words = min(len(words), self.max_seq_len[0])
			for t, token_index in enumerate(words[-n_words:]):
				data_src[i, t] = token_index

			words = tgt.split()
			n_words = min(len(words), self.max_seq_len[1])
			for t, token_index in enumerate(words[-n_words:]):
				data_tgt[i, t] = token_index

			labels.append(float(label))
			i += 1
			if i == max_n:
				break

		return data_src[:i, :], data_tgt[:i, :], np.asarray(labels)

