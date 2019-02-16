import numpy as np
import os, queue, datetime, time, socket, argparse, shutil


SOS_token = '_SOS_'
EOS_token = '_EOS_'
UNK_token = '_UNK_'

hostname = socket.gethostname()
if hostname in ['MININT-3LHNLKS', 'xiag-0228']:
	fld_data = 'd:/data'
	PLOT = True
elif 'GCR' in hostname:
	fld_data = 'data'
	PLOT = False


def makedirs(fld):
	if not os.path.exists(fld):
		os.makedirs(fld)


def line_generator(path, i_max=None):
	f = open(path, 'r', encoding='utf-8', errors='ignore')
	for line in f:
		yield line.strip('\n')
	"""
	i = 0
	line_prev = None
	while i_max is None or i < i_max:
		i += 1
		try:
			line = f.readline()
		except UnicodeDecodeError as e:
			print(e)
			print('at line: %i of %s'%(i, path))
			print('@'*20)
			print('last readable line: %s'%line_prev)
			raise
		if len(line) == 0:
			break
		yield line.strip('\n')
		line_prev = line
		"""


def write_log(path, s, PRINT=True, mode='a'):
	if PRINT:
		print(s)
	if not s.endswith('\n'):
		s += '\n'
	while True:
		try:
			with open(path, mode) as f:
				f.write(s)
			break
		except:# PermissionError as e:
			#print(e)
			print('sleeping...')
			time.sleep(2)

def load_vocab(path):
	# different with other tasks: UNK not used
	with open(path, encoding='utf-8') as f:
		lines = f.readlines()

	index2token = dict()
	token2index = dict()
	for i, line in enumerate(lines):
		token = line.strip('\n').strip()
		index2token[i + 1] = token
		token2index[token] = i + 1

	assert(SOS_token in token2index)
	assert(EOS_token in token2index)
	return index2token, token2index