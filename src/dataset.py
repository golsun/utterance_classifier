from util import *

class Dataset:

	def __init__(self, 
		fld_data, 
		max_seq_len=[150,32],
		include_punc=False,
		):

		self.path_vocab = os.path.join(fld_data, 'vocab.txt')
		self.path_train = os.path.join(fld_data, 'train.num')
		self.path_vali = os.path.join(fld_data, 'vali.num')
		self.path_test = os.path.join(fld_data, 'test.num')

		self.include_punc = include_punc

		# load token dictionary

		self.max_seq_len = max_seq_len
		self.index2token, self.token2index = load_vocab(self.path_vocab)

		self.SOS = self.token2index[SOS_token]
		self.EOS = self.token2index[EOS_token]
		self.num_tokens = len(self.token2index)	# not including 0-th
		
		if not self.include_punc:
			self.is_word = dict()
			for ix in self.index2token:
				self.is_word[ix] = is_word(self.index2token[ix])

		# load source-target pairs, tokenized
		
		self.reset('train')
		self.reset('vali')
		self.reset('test')


	def reset(self, task):
		self.generator = {
			'train': line_generator(self.path_train),
			'vali': line_generator(self.path_vali),
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
			if self.include_punc or is_word(token):		# skip punctuation if necessary
				seq.append(self.token2index.get(token, ix_unk))
		return seq


	def load_data(self, task, max_n, prefix):

		if 'src' in prefix:
			data_src = np.zeros((max_n, self.max_seq_len[0]))
		if 'tgt' in prefix:
			data_tgt = np.zeros((max_n, self.max_seq_len[1]))
		labels = []
		i = 0
		for line in self.generator[task]:
			if line == '':
				break
			if 'src' in prefix and 'tgt' in prefix:
				src, tgt, label = line.strip('\n').split('\t')
			else:
				tgt, label = line.strip('\n').split('\t')
			
			if 'src' in prefix:
				words = src.split()
				n_words = min(len(words), self.max_seq_len[0])
				t = 0
				for token_index in words[-n_words:]:
					token_index = int(token_index)
					if self.include_punc or self.is_word[token_index]:		# skip punctuation if necessary
						data_src[i, t] = token_index
						t += 1

			if 'tgt' in prefix:
				words = tgt.split()
				n_words = min(len(words), self.max_seq_len[1])
				t = 0
				for token_index in words[-n_words:]:
					token_index = int(token_index)
					if self.include_punc or self.is_word[token_index]:		# skip punctuation if necessary
						data_tgt[i, t] = token_index
						t += 1

			labels.append(float(label))
			i += 1
			if i == max_n:
				break

		if 'src' in prefix and 'tgt' in prefix: 
			return data_src[:i, :], data_tgt[:i, :], np.asarray(labels)
		else:
			return data_tgt[:i, :], np.asarray(labels)




def is_word(token):
	for c in token:
		if c.isalpha():
			return True
	return False


	
