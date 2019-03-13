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
			if self.include_punc or is_word(token):		# skip non-word (symbol) is necessary
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
					if self.include_punc or self.is_word[token_index]:		# skip non-word (symbol) is necessary
						data_src[i, t] = token_index
						t += 1

			if 'tgt' in prefix:
				words = tgt.split()
				n_words = min(len(words), self.max_seq_len[1])
				t = 0
				for token_index in words[-n_words:]:
					if self.include_punc or self.is_word[token_index]:		# skip non-word (symbol) is necessary
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



def mix_shuffle(path_T, path_F, fld_out, n=2e6, prob_T=0.5, repeat=False, tgt_only=False):
	# mix data with label 1 (True, path_T) and label 0 (False, path_F)
	#import numpy as np
	m_vali_test = 1000
	makedirs(fld_out)

	paths = [path_F, path_T]
	ff = [open(path, encoding='utf-8') for path in paths]
	for sub in ['vali','test','train']:
		open(fld_out+'/'+sub, 'w')

	repeats = [0, 0]
	m = [0, 0]
	sum_m = 0
	lines = []
	while sum(m) < n:
		if sum_m < m_vali_test * 2 + 10:
			prob = 0.5
		else:
			prob = prob_T
		label = int(np.random.random() < prob)
		line = ff[label].readline()
		if line == '':	# end of file
			if repeat:
				ff[label] = open(paths[label], encoding='utf-8')	# read again
				repeats[label] += 1
				print('repeat ff[%i]'%label)
			else:
				break
		m[label] += 1
		line = line.strip('\n')
		if tgt_only:
			line = line.split('\t')[-1]
		lines.append(line + '\t%i'%label)

		sum_m = sum(m)
		if sum_m % m_vali_test == 0:
			if sum_m == m_vali_test:
				sub = 'vali'
			elif sum_m == m_vali_test * 2:
				sub = 'test'
			else:
				sub = 'train'
			if sum_m % 1e4 == 0 or sub != 'train':
				print('F %.3f, T %.3f, total %.3f, prob = %.2f, writing to %s'%(m[0]/1e6, m[1]/1e6, sum_m/1e6, prob, sub))
			with open(fld_out+'/'+sub,'a',encoding='utf-8') as f:
				f.write('\n'.join(lines) + '\n')
			lines = []
	
	print('finally, repeats = %s'%repeats)
	print('F %.3f, T %.3f, total %.3f'%(m[0]/1e6, m[1]/1e6, sum_m/1e6))
	with open(fld_out+'/train', 'a',encoding='utf-8') as f:
		f.write('\n'.join(lines))
		

def txt2num(path_txt, path_vocab, tgt_only):
	path_out = path_txt + '.num'
	token2ix = dict()
	for i, line in enumerate(open(path_vocab, encoding='utf-8')):
		token2ix[line.strip('\n')] = i+1
	ix_unk = token2ix['_UNK_']
	
	open(path_out, 'w')
	lines = []
	n = 0
	for line in open(path_txt, encoding='utf-8'):
		ss = line.strip('\n').split('\t')
		if tgt_only:
			tgt, label = ss
		else:
			src, tgt, label = ss
			src_num = ' '.join([str(token2ix.get(w, ix_unk)) for w in src.split()])
		tgt_num = ' '.join([str(token2ix.get(w, ix_unk)) for w in tgt.split()])
		if tgt_only:
			lines.append(tgt_num + '\t' + label)
		else:
			lines.append(src_num + '\t' + tgt_num + '\t' + label)
		n += 1
		
		if len(lines) % 1e5 == 0:
			print('processed %.1f M lines'%(n/1e6))
			with open(path_out, 'a', encoding='utf-8') as f:
				f.write('\n'.join(lines) + '\n')
			lines = []
	
	print('processed %.1f M lines'%(n/1e6))
	with open(path_out, 'a', encoding='utf-8') as f:
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


def vocab_intersect(path_A, path_B, path_out):
	vocabs = []
	for path in [path_A, path_B]:
		print('reading '+path)
		vocab = [line.strip('\n') for line in open(path, encoding='utf-8')]
		print('len = %i'%len(vocab))
		vocabs.append(vocab)
	intersect = set(vocabs[0]) & set(vocabs[1])
	print('intersect %i'%len(intersect))
	vv = []
	for v in vocabs[0]:
		if v in intersect:
			vv.append(v)
	with open(path_out, 'w', encoding='utf-8') as f:
		f.write('\n'.join(vv))


def len_based_sample(path, wt, crit_len=30, n_max=-1):
	sum_l = 0
	sum_n = 0
	lines = []
	path_out = path+'.len%.2f'%wt
	open(path_out, 'w')
	for i, line in enumerate(open(path, encoding='utf-8')):
		tgt = line.strip('\n').split('\t')[-1]
		l = len(tgt.split())
		p = 1. - wt * (1 - min(l, crit_len)/crit_len) 
		# l = 30, p = 1
		# l = 10, p = 1 - wt * 2/3
		if np.random.random() <= p:
			lines.append(tgt)
			sum_l += l
			sum_n += 1
			if sum_n % 1e5 == 0:
				print('avg len %.3f, picked %.2f M from %.2f M'%(
					sum_l/sum_n, sum_n/1e6, i/1e6))
				with open(path_out, 'a', encoding='utf-8') as f:
					f.write('\n'.join(lines) + '\n')
				lines = []
			if sum_n == n_max:
				break
	with open(path_out, 'a', encoding='utf-8') as f:
		f.write('\n'.join(lines) + '\n')


def score_based_sample(path, wt, crit_score=0.5, n_max=-1):
	sum_s = 0
	sum_n = 0
	n_hi = 0
	lines = []
	path_out = path+'.scorewt%.2f'%wt
	open(path_out, 'w')
	for i, line in enumerate(open(path, encoding='utf-8')):
		tt = line.strip('\n').split('\t')
		if len(tt) != 4:
			continue
		src, tgt, _, score = tt
		score = float(score)
		p = 1. - wt * (1 - min(score, crit_score)/crit_score) 
		if np.random.random() <= p:
			lines.append(src + '\t' + tgt)
			sum_s += score
			sum_n += 1
			n_hi += score >= crit_score
			if sum_n % 1e5 == 0:
				print('avg score %.3f, picked %.2f M from %.2f M, hi_ratio = %.3f'%(
					sum_s/sum_n, sum_n/1e6, i/1e6, n_hi/sum_n))
				with open(path_out, 'a', encoding='utf-8') as f:
					f.write('\n'.join(lines) + '\n')
				lines = []
			if sum_n == n_max:
				break
	with open(path_out, 'a', encoding='utf-8') as f:
		f.write('\n'.join(lines) + '\n')


def is_word(token):
	for c in token:
		if c.isalpha():
			return True
	return False


if __name__ == "__main__":
	"""
	path_scored = 'D:/data/reddit/scored36M/scored.tsv'
	tlike_score = 0.37
	prob_rand = 5./36.2
	build_mixed_dataset(path_scored, tlike_score, prob_rand, 5e6, 5e6)
	"""
	"""
	path_T = 'D:/data/fuse/Holmes/combined.txt'
	path_F = 'D:/data/reddit/out(d2-10, l30w, s1, t0)/train.txt'

	fld_out = 'd:/data/classifier/reddit1f_holmes2_probT0.1'
	mix_shuffle(path_T, path_F, fld_out, n=6e6, prob_T=0.1, tgt_only=True)
	#"""
	
	"""
	for sub in ['vali','test','train']:
		path_txt = fld_out + '/' + sub
		path_vocab = fld_out + '/vocab.txt'
		txt2num(path_txt, path_vocab, tgt_only=True)
		#"""
	"""
	path_A = 'D:/data/fuse/holmes/vocab.txt'
	path_B = 'D:/data/reddit/out(d2-10, l30w, s0, t1)/ref_3/vocab.txt'
	path_out = 'd:/data/classifier/reddit3f_holmes2_probT0.1/vocab.txt'
	vocab_intersect(path_A, path_B, path_out)
	#"""

	"""
	path = 'D:/data/reddit/out(d2-10, l30w, s0, t1)/ref_3/train.txt'
	for part in range(5):
		print('part %i'%part)
		score_based_sample(
			'D:/data/reddit/out(d2-10, l30w, s1, t0)/2012/train.txt.part%i.scored'%part, 
			wt=1., n_max=3e6)
			"""
	
	print(is_word('.'))
	print(is_word("n't"))
	print(is_word("wtf"))
	
	
