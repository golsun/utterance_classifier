"""
Twitter vs. Reddit classifier
"""
from keras.models import Model
from keras.layers import Input, GRU, Dense, Embedding, Dropout, Concatenate, Lambda, Add, Subtract, Multiply, GaussianNoise
from keras.utils import plot_model
from keras.callbacks import Callback
from keras.optimizers import Adam
from dataset import Dataset
from util import *




class LossHistory(Callback):
	def reset(self):
		self.losses = []

	def on_train_begin(self, logs={}):
		self.reset()

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))




class Classifier():

	def __init__(self, fld, dataset, encoder_depth, rnn_units, mlp_depth, mlp_units, tgt_only=False, lr=1e-4, dropout=0.):
		if tgt_only:
			self.prefix = ['tgt']
		else:
			self.prefix = ['src','tgt']
		self.history = LossHistory()

		self.fld = fld
		self.dataset = dataset
		self.encoder_depth = encoder_depth
		self.rnn_units = rnn_units
		self.mlp_depth = mlp_depth
		self.mlp_units = mlp_units
		self.lr = lr
		self.dropout = dropout
		makedirs(self.fld + '/epochs')

		self.log_train = self.fld + '/train'

		self.dataset.reset('test')
		self.vali_data = self.load_data('vali', 2000)
		self.dataset.reset('test')


	def load_weights(self):
		self.model.load_weights(self.fld + '/model.h5')


	def _create_layers(self):
		layers = dict()

		layers['embedding'] = Embedding(
				self.dataset.num_tokens + 1,		# +1 as mask_zero 
				self.rnn_units, mask_zero=True, 
				name='embedding')

		for prefix in self.prefix:
			for i in range(self.encoder_depth):
				name = '%s_encoder_rnn_%i'%(prefix, i)
				layers[name] = GRU(
						self.rnn_units, 
						return_state=True,
						return_sequences=True, 
						name=name)

		for i in range(self.mlp_depth - 1):
			name = 'mlp_%i'%i
			layers[name] = Dense(
				self.mlp_units, 
				activation='tanh', name=name)

		name = 'mlp_%i'%(self.mlp_depth - 1)
		layers[name] = Dense(1, activation='sigmoid', name=name)
		return layers


	def _stacked_rnn(self, rnns, inputs, initial_states=None):
		if initial_states is None:
			initial_states = [None] * len(rnns)

		outputs, state = rnns[0](inputs, initial_state=initial_states[0])
		states = [state]
		for i in range(1, len(rnns)):
			outputs, state = rnns[i](outputs, initial_state=initial_states[i])
			states.append(state)
		return outputs, states


	def _build_encoder(self, inputs, layers, prefix):
		_, encoder_states = self._stacked_rnn(
				[layers['%s_encoder_rnn_%i'%(prefix, i)] for i in range(self.encoder_depth)], 
				layers['embedding'](inputs))
		latent = encoder_states[-1]
		return latent


	def build_model(self):
		layers = self._create_layers()

		encoder_inputs = dict()
		latents = []
		for prefix in self.prefix:
			encoder_inputs[prefix] = Input(shape=(None,), name=prefix+'_encoder_inputs')
			latents.append(self._build_encoder(encoder_inputs[prefix], layers, prefix=prefix))

		if len(self.prefix) > 1:
			out = Concatenate()(latents)
			inp = [encoder_inputs['src'], encoder_inputs['tgt']]
		else:
			out = latents[0]
			inp = encoder_inputs[self.prefix[0]]
		out = Dropout(self.dropout)(out)
		for i in range(self.mlp_depth):
			out = layers['mlp_%i'%i](out)

		self.model = Model(inp, out)
		self.model.compile(optimizer=Adam(lr=self.lr), loss='binary_crossentropy')
		if PLOT:
			plot_model(self.model, self.fld + '/model.png',	show_shapes=True)


	def save_weights(self, fname, subfld=False):
		fname += '.h5'
		if subfld:
			path = os.path.join(self.fld, 'epochs', fname)
		else:
			path = os.path.join(self.fld, fname)
		while True:
			try:
				self.model.save_weights(path)
				break
			except IOError as e:
				print(e)
				time.sleep(2)
		print('saved to: '+path)


	def train(self, 
		epochs=1, 
		batch_size=128, 
		batch_per_load=100):

		shutil.copyfile(self.dataset.path_vocab, self.fld + '/vocab.txt')

		self.n_trained = 0
		for epoch in range(epochs):
			self.dataset.reset('train')
			while True:
				s = '\n***** Epoch %i/%i, trained %.2fM *****'%(
					epoch + 1, epochs, self.n_trained/1e6)
				write_log(self.log_train + '.log', s)
				m = self.train_a_load(batch_size, batch_per_load)
				if m == 0:
					break
				self.save_weights('epoch%i'%(epoch + 1), subfld=True)
		self.save_weights('model')


	def test(self):
		print('testing...')
		inp, labels = self.load_data('test', 10000)
		loss_vali = self.model.evaluate(inp, labels, verbose=0)
		labels_pred = self.model.predict(inp, verbose=0).ravel()
		acc = sum(labels == 1.*(labels_pred > 0.5))/len(labels)
		print('loss = %.2f, accuracy = %.2f'%(loss_vali, acc))

		lines = ['\t'.join(['turn','src','tgt','truth','pred'])]
		for i in range(len(labels)):
			if len(self.prefix) == 1:
				line = '\t'.join([
					'0',
					self.dataset.seq2text(inp[i,:]),
					'%i'%labels[i],
					'%.4f'%labels_pred[i],
					])
			else:
				data_src, data_tgt = inp
				txt_src = self.dataset.seq2text(data_src[i,:])
				line = '\t'.join([
					'%i'%(len(txt_src.split('EOS')) + 1),
					txt_src,
					self.dataset.seq2text(data_tgt[i,:]),
					'%i'%labels[i],
					'%.4f'%labels_pred[i],
					])
			lines.append(line)

		fld_vali = self.fld + '/post'
		makedirs(fld_vali)
		with open(fld_vali + '/test.tsv', 'w', encoding='utf-8') as f:
			f.write('\n'.join(lines))


	def interact(self):
		while True:
			print('--- context ---')
			src = input()
			if len(src) == 0:
				break
			print('--- response ---')
			tgt = input()
			if len(tgt) == 0:
				break

			data_src = np.zeros((1, self.dataset.max_seq_len[0]))
			data_tgt = np.zeros((1, self.dataset.max_seq_len[1]))

			words = dataset.text2seq(src)
			n_words = min(len(words), self.dataset.max_seq_len[0])
			for t, token_index in enumerate(words[-n_words:]):
				data_src[0, t] = token_index

			words = dataset.text2seq(tgt)
			n_words = min(len(words), self.dataset.max_seq_len[1])
			for t, token_index in enumerate(words[-n_words:]):
				data_tgt[0, t] = token_index

			print('--- score ---\n%.4f'%(self.model.predict([data_src, data_tgt], verbose=0)))


	def load_data(self, sub, size):
		data = self.dataset.load_data(sub, size, prefix=self.prefix)
		if len(self.prefix) == 1:
			data_tgt, labels = data
			inp = data_tgt
		else:
			data_src, data_tgt, labels = data
			inp = [data_src, data_tgt]
		return inp, labels



	def train_a_load(self, batch_size, batch_per_load):
		inp, labels = self.load_data('train', batch_size * batch_per_load)
		m = len(labels)
		if m == 0:
			return 0
		
		t0 = datetime.datetime.now()
		t0_str = str(t0).split('.')[0]
		s = '\nstart:\t%s'%t0_str
		write_log(self.log_train + '.log', s)

		self.model.fit(
			inp, 
			labels,
			batch_size=batch_size,
			callbacks=[self.history])

		dt = (datetime.datetime.now() - t0).seconds
		loss = np.mean(self.history.losses)

		# vali --------------------

		inp, labels = self.vali_data
		loss_vali = self.model.evaluate(inp, labels, verbose=0)
		labels_pred = self.model.predict(inp, verbose=0)
		acc = sum(labels.ravel() == 1.*(labels_pred.ravel() > 0.5))/len(labels)

		ss = [
			'spent:    %i sec'%dt,
			'train:    %.4f'%loss,
			'vali:     %.4f'%loss_vali,
			'vali_acc: %.4f'%acc,
			]
		s = '\n'.join(ss)
		
		self.n_trained += m
		write_log(self.log_train + '.log', s)
		write_log(self.log_train + '.tsv', 
			'\t'.join(['%.2f'%(self.n_trained/1e6), '%i'%dt, '%.4f'%loss, '%.4f'%loss_vali, '%.4f'%acc]), 
			PRINT=False)

		return m



def post(fld):
	import matplotlib.pyplot as plt
	grouped_by_true = {'all':[[], []]}
	for line in line_generator(fld + '/test.tsv'):
		ss = line.split('\t')
		turn = ss[0]
		truth = ss[-2]
		pred = ss[-1]
		try:
			truth = int(truth)
		except ValueError:
			continue
		pred = float(pred)
		grouped_by_true['all'][truth].append(pred)
		if turn not in grouped_by_true:
			grouped_by_true[turn] = [[],[]]
		grouped_by_true[turn][truth].append(pred)

	path = fld + '/test_score_vs_turn.tsv'
	with open(path,'w') as f:
		f.write('turn\tReddit\tTwitter\n')
	for k in ['all']:#sorted(grouped_by_true.keys()):
		reddit = grouped_by_true[k][0]
		twitter = grouped_by_true[k][1]
		f, ax = plt.subplots()
		ax.hist(reddit, color='b', label='Reddit', alpha=0.5, bins=30)
		ax.hist(twitter, color='r', label='Twitter', alpha=0.5, bins=30)
		ax.set_title('%s-turn: Reddit = %.2f, Twitter = %.2f'%(k, np.mean(reddit), np.mean(twitter)))
		ax.set_xlabel('score')
		y = ax.get_ylim()[1] * 0.5
		ax.text(0.2, y, 'False', color='b')
		ax.text(0.6, y, 'True', color='r')
		plt.savefig(fld + '/%s_hist.png'%k)
		plt.close()

		n = len(reddit)
		plt.plot([100.*i/n for i in range(n)], [s for s in reversed(sorted(reddit))])
		plt.xlabel('rank (%)')
		plt.ylabel('score')
		plt.title('Reddit')
		plt.savefig(fld + '/%s_rank.png'%k)
		plt.close()



def cal_score(classifier, path_in):
	path_out = path_in + '.scored'
	print('scoring '+path_in)
	batch_size = 100
	open(path_out, 'w')
	lines = open(path_in, encoding='utf-8').readlines()
	sum_score = 0.
	n_high = 0

	def get_tensor():
		data_src = np.zeros((batch_size, classifier.dataset.max_seq_len[0]))
		data_tgt = np.zeros((batch_size, classifier.dataset.max_seq_len[1]))
		return data_src, data_tgt

	j = 0
	data_src, data_tgt = get_tensor()
	src_tgt = []

	n_high = 0
	n = 0
	sum_score = 0.

	for line in open(path_in, encoding='utf-8'):
		n += 1
		src_tgt.append(line.strip('\n'))
		src, tgt = line.strip('\n').split('\t')
		src = src.split(' EOS ')[-1].strip()		# as the classifier is only trained on 2-turn data
		seq_src = classifier.dataset.text2seq(src)
		seq_tgt = classifier.dataset.text2seq(tgt)

		n_words = min(len(seq_src), classifier.dataset.max_seq_len[0])
		for t, token_index in enumerate(seq_src[-n_words:]):
			data_src[j, t] = token_index

		n_words = min(len(seq_tgt), classifier.dataset.max_seq_len[1])
		for t, token_index in enumerate(seq_tgt[-n_words:]):
			data_tgt[j, t] = token_index

		j += 1
		if j == batch_size:
			scores = classifier.model.predict([data_src, data_tgt], verbose=0).ravel()
			sum_score += np.sum(scores)
			lines = []
			for i in range(batch_size):
				score = scores[i]
				lines.append(src_tgt[i] + '\t' + '%.4f'%score)
				n_high += score > 0.5
			with open(path_out, 'a', encoding='utf-8') as f:
				f.write('\n'.join(lines) + '\n')
			print('processed %.3f M, avg_score = %.2f, perc > 0.5 = %.1f'%(
					n/1e6,
					sum_score/n,
					n_high/n*100.,
				))
			
			# reset
			j = 0
			data_src, data_tgt = get_tensor()
			src_tgt = []

					




	

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', default='score')
	parser.add_argument('--encoder_depth', type=int, default=2)
	parser.add_argument('--rnn_units', type=int, default=32)
	parser.add_argument('--mlp_depth', type=int, default=2)
	parser.add_argument('--mlp_units', type=int, default=32)
	parser.add_argument('--tgt_only', action='store_true')
	parser.add_argument('--data_name', default='reddit3f_holmes2_probT0.1')		# for training
	parser.add_argument('--score_path', default='')
	args = parser.parse_args()

	fld = 'out/en(%i,%i),mlp(%i,%i),tgt_only%i'%(
		args.encoder_depth, args.rnn_units, args.mlp_depth, args.mlp_units, args.tgt_only)
	
	if args.mode == 'score':
		fld_vocab = fld
	else:
		fld_vocab = fld_data + '/' + args.data_name
	dataset = Dataset(fld_vocab)

	classifier = Classifier(fld, dataset, 
		args.encoder_depth, args.rnn_units, args.mlp_depth, args.mlp_units, tgt_only=args.tgt_only)
	classifier.build_model()
	if args.mode != 'train':
		classifier.load_weights()
	if args.mode in ['train', 'continue']:
		classifier.train()
	elif args.mode == 'interact':
		classifier.interact()
	elif args.mode == 'test':
		classifier.test()
		if PLOT:
			post(fld + '/post')
	elif args.mode == 'post':
		post(fld + '/post')
	elif args.mode == 'score':
		cal_score(classifier, arg.score_path)



