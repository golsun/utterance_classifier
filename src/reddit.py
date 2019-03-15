#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sean Xiang Gao based on Michel Galley's script create_official_data.py for DSTC-task2
* 	no facts (thus no conversation ID, domain filter, craw func...)
* 	no score filter (but will store score)
* 	changed len limit (either by words or by chars, check all utterance indepently)
*	changed get_convo, don't add submission at the head when not necessary
*	changed norm_sentence
*	norm_sentence before saving pickle (so don't norm one sentence duplicately)
"""

import sys
import time
import os.path
import re
import argparse
import traceback
import json
import bz2
#import pickle
#import markdown
from bs4 import BeautifulSoup
from util import *

def makedirs(fld):
	if not os.path.exists(fld):
		os.makedirs(fld)

PICKLE_MAX_LEN = 1e4
TAG_COMMENT = 't1_'
TAG_SUBMISSION = 't3_'
dontuse = '__dontuse__'
url_str = '__url__'

parser = argparse.ArgumentParser()

parser.add_argument("dump_name", help="YYYY-MM, dumped files to be loaded")
parser.add_argument("--max_len", default=30, type=int)
parser.add_argument("--max_len_type", default='w')		# w for words, c for chars
parser.add_argument("--min_depth", default=2, type=int)
parser.add_argument("--max_depth", default=10, type=int)
parser.add_argument("--min_score", default=0, type=int)
parser.add_argument("--use_title", default=1, type=int)
parser.add_argument("--split_size", default=int(5e5), type=int)
parser.add_argument("--task", default='conv')


#parser.add_argument("--nsubmissions", help="Number of submissions to process (< 0 means all)", default=-1, type=int)
#parser.add_argument("--ncomments", help="Number of comments to process (< 0 means all)", default=-1, type=int)



args = parser.parse_args()

fields_subm = [ "id", "subreddit", "score", "num_comments", "domain", "permalink", "title" ]
fields_comm = [ "id", "author", "parent_id", "link_id", "score", "n_char", "body"]


def get_submission_id(submission):
	return TAG_SUBMISSION + submission["id"]
def get_comment_id(comment):
	return TAG_COMMENT + comment["id"]


def norm_sentence(txt):
	txt = txt.lower()

	# url and tag
	words = []
	for word in txt.lower().split():
		if word[0] == '#':	# don't allow tag
			continue
		i = word.find('http') 
		if i >= 0:
			word = word[:i] + ' ' + '__url__'
		words.append(word.strip())
	txt = ' '.join(words)

	# remove illegal char
	txt = txt.replace(chr(92),'')	# chr(92) = '\'. as twitter has 'b\/c' rather than 'b/c'
	txt = txt.replace("b/c","because").replace('j/k','just kidding').replace('w/o','without').replace('w/','with')
	txt = re.sub('__mention__','MENTION',txt)
	txt = re.sub('__url__','URL',txt)
	txt = re.sub(r"[^A-Za-z0-9():,.!?'“” ]", " ", txt)
	txt = re.sub('MENTION','__mention__',txt)	
	txt = re.sub('URL','__url__',txt)	

	# contraction
	add_space = ["'s", "'m", "'re", "n't", "'ll","'ve","'d","'em"]
	tokenizer = TweetTokenizer(preserve_case=False)
	txt = ' ' + ' '.join(tokenizer.tokenize(txt)) + ' '
	txt = txt.replace(" won't ", " will n't ")
	txt = txt.replace(" can't ", " can n't ")
	for a in add_space:
		txt = txt.replace(a+' ', ' '+a+' ')
	
	# remove un-necessary space
	return ' '.join(txt.split())


def reddit_norm_sentence(txt):
	txt = txt.lower().replace('r/','')
	return norm_sentence(txt)



def extract_submissions(fld_bz2, fld_split, size=2e5):
	# filter by
	#	1. num_comments >= 2

	path_in = fld_bz2 + '/RS_%s.bz2'%args.dump_name

	n = 0
	m = 0
	sub = 0
	sid = []
	sids = []
	lines = []
	with bz2.open(path_in, 'rt', encoding="utf-8") as f:
		for line in f:
			n += 1
			if n%1e4 == 0:
				print('[%s] selected %.3fM from %.2fM submissions'%(
					args.dump_name, m/1e6, n/1e6))
			try:
				submission = json.loads(line)
				if int(submission['num_comments']) < 2:		# filter 1
					continue
				submission['title'] = reddit_norm_sentence(submission['title'])
				lines.append('\t'.join([str(submission[k]) for k in fields_subm]))
				m += 1
				sid.append(get_submission_id(submission))

			except Exception:
				traceback.print_exc()
				continue

			if len(sid) == size:
				print('writing submissions_sub%i'%sub)
				sids.append(set(sid))
				with open(fld_split + '/rs_sub%i.tsv'%sub, 'w', encoding='utf-8') as f:
					#f.write('\t'.join(fields_subm) + '\n')
					f.write('\n'.join(lines))
				sid = []
				lines = []
				sub += 1

	print('writing submissions_sub%i'%sub)
	sids.append(set(sid))
	with open(fld_split + '/rs_sub%i.tsv'%sub, 'w', encoding='utf-8') as f:
		#f.write('\t'.join(fields_subm) + '\n')
		f.write('\n'.join(lines))
	print('extract_submissions done.\n')
	return sids, m, n


def extract_comments(fld_bz2, fld_split, sids):
	# filter by
	# 	1. not deleted
	#	2. len > 1
	# 	3. no comment in line

	path_in = fld_bz2 + '/RC_%s.bz2'%args.dump_name

	n = 0
	m = 0
	n_sub = len(sids)
	lines = [[] for i in range(n_sub)]
	for sub in range(n_sub):
		open(fld_split + '/rc_sub%i.tsv'%sub, 'w')

	with bz2.open(path_in, 'rt', encoding="utf-8") as f:
		for line in f:
			n += 1
			if n%1e4 == 0:
				print('[%s] selected %.3fM from %.2fM comments'%(
					args.dump_name, m/1e6, n/1e6))

				for sub in range(n_sub):
					print('    sub %i: %i'%(sub, len(lines[sub])))
					if len(lines[sub]) > 0:
						with open(fld_split + '/rc_sub%i.tsv'%sub, 'a', encoding='utf-8') as f:
							f.write('\n'.join(lines[sub]) + '\n')
						lines[sub] = []
			try:
				comment = json.loads(line)
				if comment['body'] == '[deleted]':			# filter 1
					continue
				if '>' in comment['body'] or '&gt;' in comment['body']:		# filter 3: '&gt;' means '>'
					continue
				sid = comment['link_id']
				for sub in range(n_sub):
					if sid in sids[sub]:
						comment['n_char'] = len(comment['body'])
						comment['body'] = reddit_norm_sentence(comment['body'])
						if len(comment['body'].split()) < 2:	# filter 2
							break
						lines[sub].append('\t'.join([str(comment[k]) for k in fields_comm]))
						m += 1
						break

			except Exception:
				traceback.print_exc()

	print('the rest...')
	for sub in range(n_sub):
		print('    sub %i: %i'%(sub, len(lines[sub])))
		with open(fld_split + '/rc_sub%i.tsv'%sub, 'a', encoding='utf-8') as f:
			f.write('\n'.join(lines[sub]))

	print('extract_comments done.\n')
	return m, n




def get_convo(cid, submissions, comments, depth=args.max_depth):
	if depth == 0:
		return []
	c = comments[cid]
	if args.max_len_type == 'w' and len(c['body'].split()) > args.max_len:	# len filter
		return []
	if args.max_len_type == 'c' and int(c['n_char']) > args.max_len:
		return []

	pid = c['parent_id']
	if args.use_title and pid.startswith(TAG_SUBMISSION):
		txts = [ submissions[c['link_id']]['title'] ]
	elif pid in comments:
		txts = get_convo(pid, submissions, comments, depth-1)
	else:
		txts = []
	txts.append(c['body'])
	return txts



def save_convo(path_rs, path_rc, path_out):
	# filter by
	# 	1. score
	# 	2. len
	# 	3. depth

	print('reading submissions...')
	submissions = dict()
	with open(path_rs, encoding='utf-8') as f:
		for line in f:
			cells = line.strip('\n').strip().split('\t')
			try:
				submission = dict([(fields_subm[i], cells[i]) for i in range(len(fields_subm))])
			except Exception:
				#traceback.print_exc()
				continue
			submissions[get_submission_id(submission)] = submission

	print('reading comments...')
	comments = dict()
	with open(path_rc, encoding='utf-8') as f:
		for line in f:
			cells = line.strip('\n').strip().split('\t')
			try:
				comment = dict([(fields_comm[i], cells[i]) for i in range(len(fields_comm))])
			except Exception:
				traceback.print_exc()
				continue
			comments[get_comment_id(comment)] = comment

	sorted_id = sorted([(
					comments[cid]['link_id'],
					comments[cid]['parent_id'],
					cid
					) for cid in comments])

	n = len(comments)
	print('total comments: %i'%n)

	i = 0
	m = 0
	lines = []
	sum_resp_len = 0

	for sid, pid, cid in sorted_id:
		i += 1
		if i%1e5 == 0:
			print('selected %.2fM from %.1f/%.1fM comments'%(m/1e6, i/1e6, n/1e6))
			if len(lines) > 0:
				with open(path_out, 'a', encoding="utf-8") as f:
					f.write('\n'.join(lines) + '\n')
			lines = []

		comment = comments[cid]
		score = int(comment['score'])
		if score < args.min_score:		# filter 1
			continue
		try:
			txts = get_convo(cid, submissions, comments)	# filter 2
		except Exception:
			#print('vvvvvvvvvvvv handled exception vvvvvvvvvv')
			#traceback.print_exc()
			#print('^^^^^^^^^^^^ handled exception ^^^^^^^^^^')
			continue
		if len(txts) < args.min_depth:				# filter 3
			continue	

		lines.append(join_contexts(txts[:-1]) + '\t' + txts[-1])
		sum_resp_len += len(txts[-1].split())
		m += 1		

	avg_len = sum_resp_len/m
	with open(path_out, 'a', encoding="utf-8") as f:
		f.write('\n'.join(lines) + '\n')
	print('finally selected %i/%i, avg len = %.2f'%(m, n, avg_len))
	return m, n, avg_len




def extract():
	makedirs(fld_split)
	sids, ms, ns = extract_submissions(fld_root + '/bz2', fld_split, size=args.split_size)
	mc, nc = extract_comments(fld_root + '/bz2', fld_split, sids)
	with open(fld_split + '/stat.tsv', 'a') as f:
		f.write('\t'.join(map(str, [args.dump_name, mc, nc, ms, ns])) + '\n')


def build_conv():
	fld_out = fld_root + '/out(d%i-%i, l%i%s, s%i, t%i)'%(
			args.min_depth, args.max_depth, args.max_len, args.max_len_type, args.min_score, args.use_title)
	makedirs(fld_out)
	path_out = fld_out + '/%s.tsv'%args.dump_name
	print(path_out)

	sub = 0
	sum_m = 0
	sum_n = 0
	while True:
		path_rs = fld_split + '/rs_sub%i.tsv'%sub
		if not os.path.exists(path_rs):
			print('no such file: '+path_rs)
			break
		print('-'*10 + ' sub%i '%sub + '-'*10)
		path_rc = path_rs.replace('/rs_', '/rc_')
		m, n, avg_len = save_convo(path_rs, path_rc, path_out)
		with open(fld_out + '/stat.tsv', 'a') as f:
			f.write('\t'.join([args.dump_name, str(sub), str(m), str(n), '%.2f'%avg_len]) + '\n')
		sum_m += m
		sum_n += n
		sub += 1

	with open(fld_out + '/stat.tsv', 'a') as f:
		f.write('\t'.join([args.dump_name, 'all', str(sum_m), str(sum_n), '']) + '\n')


fld_root = 'd:/data/reddit/'
fld_split = fld_root + '/split(%.1fM)/%s'%(args.split_size/1e6, args.dump_name)
if args.task == 'extract':
	extract()
elif args.task == 'conv':
	build_conv()
