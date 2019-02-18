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



def build_mixed_dataset(path_scored, tlike_score, prob_rand, n_tlike, n_rand):

	path_out = path_scored + '.picked'
	w_score = 0.1
	update_rate = 0.1

	m_tlike = 0
	m_rand = 0
	lines = []
	sum_score_tlike = 0.
	sum_score_rand = 0.

	for sub in ['vali', 'test', 'train']:
		open(path_out + '.' + sub, 'w')
	
	for line in open(path_scored, encoding='utf-8'):
		src, tgt, score = line.strip('\n').split('\t')
		line = src + '\t' + tgt
		score = float(score)

		if m_rand < n_rand and np.random.random() < prob_rand:
			lines.append(line)
			m_rand += 1
			sum_score_rand += score
		elif score >= tlike_score:
			lines.append(line)
			m_tlike += 1
			sum_score_tlike += score
		else:
			continue

		m = (m_tlike + m_rand)
		if len(lines) == 5000:
			if m == 5000:
				sub = 'vali'
			elif m == 10000:
				sub = 'test'
			else:
				sub = 'train'

			with open(path_out + '.' + sub, 'a', encoding='utf-8') as f:
				f.write('\n'.join(lines) + '\n')
			lines = []

			if sub != 'train' or m % 1e5 == 0:
				print('wrote to %s, picked %.3f M, t-like/total = %.2f, avg_score = %.2f/%.2f'%(
					sub,
					m/1e6,
					m_tlike/m,
					sum_score_tlike/m_tlike,
					sum_score_rand/m_rand
					))

	with open(path_out + '.train', 'a', encoding='utf-8') as f:
		f.write('\n'.join(lines))
	m = (m_tlike + m_rand)
	print('finally, picked %.3f M, t-like/total = %.2f, avg_score = %.2f/%.2f=>%.2f'%(
		m/1e6,
		m_tlike/m,
		sum_score_tlike/m_tlike,
		sum_score_rand/m_rand,
		(sum_score_rand + sum_score_tlike)/(m_tlike + m_rand),
		))	


if __name__ == "__main__":
	path_scored = 'D:/data/reddit/scored36M/scored.tsv'
	tlike_score = 0.37
	prob_rand = 5./36.2
	build_mixed_dataset(path_scored, tlike_score, prob_rand, 5e6, 5e6)

