#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filter stdin input using a dictionary of common or "bland" ngrams (read from 1st argument), keeping only sentences that satisfy either of the following conditions:
(1) the current sentence contribute sat least one ngrams not in the list of bland ngrams;
(2) at least one ngram of the current sentence has been seen less than N times.
Note: SHUFFLE your data before running it through this script, otherwise output data will be biased towards data earlier in the stream!
Michel Galley mgalley@microsoft.com
"""

import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--common_ngrams", required=True, help="list of common ngrams")
parser.add_argument("--order", default=3, type=int, help="N-gram order")
parser.add_argument("--filter_col", default=1, type=int, help="Column to filter (default is the 2nd column, which is often the response in a tsv file)")
parser.add_argument("--header_col", default=-1, type=int, help="Column containing a header. If >= 0, the program adds a 'blandness' field to the header.")
parser.add_argument("--count", default=100, type=int, help="Keep sentence if any of its ngrams was seen less than COUNT times, or if any of its ngrams is not on the list.")

args = parser.parse_args()

def read_ngrams(ngram_file):
	ngrams = {}
	with open(ngram_file, encoding='utf-8') as f:
		for line in f:
			n = line.rstrip()
			#print("ngram: " + n)
			ngrams[n] = 0
	return ngrams

def filter_data(ngrams):
	N = args.order
	S = args.count
	for line in sys.stdin:
		line = line.rstrip()
		cols = line.split('\t')
		col = cols[args.filter_col]
		if args.header_col >= 0:
			header = cols[args.header_col]
		words = ('<s> ' + col + ' </s>').split()
		#print("words: " + " ".join(words))
		keep = False
		n_total = 0 # total number of ngrams
		n_bland = 0 # total number of 'bland' ngrams
		n_bland_progressive = 0 # total number of 'bland' ngrams, counting as bland only those already seen >= COUNT times
		for i in range(0, len(words)-N+1):
			n_total += 1
			n = " ".join(words[i:i+N])
			#print("ngram: " + n)
			if n not in ngrams.keys():
				keep = True
			else:
				ngrams[n] += 1
				n_bland += 1
				if ngrams[n] < S:
					keep = True
				else:
					n_bland_progressive += 1
		if args.header_col >= 0:
			h = "bland={0:.2f},bland_filter={1:.2f}".format(n_bland/n_total, n_bland_progressive/n_total)
			if header:
				cols[args.header_col] = header + "," + h
			else:
				cols[args.header_col] = h
			line = "\t".join(cols)
		if keep:
			print(line)
		else:
			print("Skip: " + line, file=sys.stderr)

if __name__ == "__main__":
	ngrams = read_ngrams(args.common_ngrams)
	filter_data(ngrams)
