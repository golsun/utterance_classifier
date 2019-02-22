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


	def load_data(self, task, max_n, prefix):

		if 'src' in prefix:
			data_src = np.zeros((max_n, self.max_seq_len[0]))
		if 'tgt' in prefix:
			data_tgt = np.zeros((max_n, self.max_seq_len[1]))
		labels = []
		i = 0
		for line in self.generator[task]:
			if 'src' in prefix and 'tgt' in prefix:
				src, tgt, label = line.strip('\n').split('\t')
			else:
				tgt, label = line.strip('\n').split('\t')
			
			if 'src' in prefix:
				words = src.split()
				n_words = min(len(words), self.max_seq_len[0])
				for t, token_index in enumerate(words[-n_words:]):
					data_src[i, t] = token_index

			if 'tgt' in prefix:
				words = tgt.split()
				n_words = min(len(words), self.max_seq_len[1])
				for t, token_index in enumerate(words[-n_words:]):
					data_tgt[i, t] = token_index

			labels.append(float(label))
			i += 1
			if i == max_n:
				break

		if 'src' in prefix and 'tgt' in prefix: 
			return data_src[:i, :], data_tgt[:i, :], np.asarray(labels)
		else:
			return data_tgt[:i, :], np.asarray(labels)



def mix_shuffle(path_T, path_F, path_out, n=2e6, repeat=False):
	# mix data with label 1 (True, path_T) and label 0 (False, path_F)
	import numpy as np
	m_vali_test = 1000

	paths = [path_F, path_T]
	ff = [open(path, encoding='utf-8') for path in paths]
	for sub in ['vali','test','train']:
		open(path_out+'.'+sub, 'w')

	repeats = [0, 0]
	m = [0, 0]
	lines = []
	while sum(m) < n:
		label = int(np.round(np.random.random()))
		line = ff[label].readline()
		if line == '':	# end of file
			if repeat:
				ff[label] = open(paths[label], encoding='utf-8')	# read again
				repeats[label] += 1
				print('repeat ff[%i]'%label)
			else:
				break
		m[label] += 1
		lines.append(line.strip('\n') + '\t%i'%label)

		sum_m = sum(m)
		if sum_m % m_vali_test == 0:
			if sum_m == m_vali_test:
				sub = 'vali'
			elif sum_m == m_vali_test * 2:
				sub = 'test'
			else:
				sub = 'train'
			if sum_m % 1e4 == 0 or sub != 'train':
				print('F %.3f, T %.3f, total %.3f, writing to %s'%(m[0]/1e6, m[1]/1e6, sum_m/1e6, sub))
			with open(path_out+'.'+sub,'a',encoding='utf-8') as f:
				f.write('\n'.join(lines) + '\n')
			lines = []
	
	print('finally, repeats = %s'%repeats)
	print('F %.3f, T %.3f, total %.3f'%(m[0]/1e6, m[1]/1e6, sum_m/1e6))
	with open(path_out+'.train','a',encoding='utf-8') as f:
		f.write('\n'.join(lines))
		

def build_mixed_dataset(path_scored, tlike_score, prob_rand, n_tlike, n_rand):
	# given a scored txt file, output mixture of half high-score and half rand-sampled

	path_out = path_scored + '.picked'

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
	"""
	path_scored = 'D:/data/reddit/scored36M/scored.tsv'
	tlike_score = 0.37
	prob_rand = 5./36.2
	build_mixed_dataset(path_scored, tlike_score, prob_rand, 5e6, 5e6)
	"""
	path_T = 'D:/data/fuse/Holmes/combined.txt'
	path_F = 'D:/data/reddit/out(d2-10, l30w, s0, t1)/ref_3/filtered/train.txt'
	path_out = 'd:/data/fuse/classifier_reddit3f+holmes2/mixed'
	mix_shuffle(path_T, path_F, path_out, n=1e6, repeat=True)
