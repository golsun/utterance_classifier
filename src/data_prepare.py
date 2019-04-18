from util import *
from collections import Counter

def mix_shuffle(path_T, path_F, fld_out, n=2e6, prob_T=0.5, repeat=False, tgt_only=True):
    # mix data with label 1 (True, path_T) and label 0 (False, path_F)
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
            if sum_m % 1e5 == 0 or sub != 'train':
                print('F %.3f, T %.3f, total %.3f, prob = %.2f, writing to %s'%(m[0]/1e6, m[1]/1e6, sum_m/1e6, prob, sub))
            with open(fld_out+'/'+sub,'a',encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
            lines = []
    
    print('finally, repeats = %s'%repeats)
    print('F %.3f, T %.3f, total %.3f'%(m[0]/1e6, m[1]/1e6, sum_m/1e6))
    with open(fld_out+'/train', 'a',encoding='utf-8') as f:
        f.write('\n'.join(lines))
        

def txt2num(path_txt, path_vocab, tgt_only=True):
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
        if np.random.random() > p:
            continue

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

    print('avg len %.3f, picked %.2f M from %.2f M'%(
        sum_l/sum_n, sum_n/1e6, i/1e6))
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



def rand_subset(path, n, tgt_only=True, p=0.5):
    lines = []
    m = 0
    for line in open(path, encoding='utf-8'):
        if np.random.random() < p:
            lines.append(line.strip('\n').split('\t')[-1])
            m += 1
            if m == n:
                break
    with open(path+'.subset', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
      
def shuffle_file(path):
    lines = open(path, encoding='utf-8').readlines()
    ii = list(range(len(lines)))
    np.random.seed(9)
    np.random.shuffle(ii)
    lines_new = [lines[i].strip('\n') for i in ii]
    with open(path+'.shuffled', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines_new))
        
def avg_len(path, write=False):
    ll = []
    ll_short = []
    lines_short = []
    for line in open(path, encoding='utf-8'):
        ww = line.strip('\n').split()
        ll.append(len(ww))
        if len(ww) < 30:
            ll_short.append(len(ww))
            lines_short.append(line.strip('\n'))
    
    print('avg len: %.2f'%np.mean(ll))
    print('greater than 30: %.2f'%(sum(np.array(ll)>30)/len(ll)))
    print('avg len short: %.2f'%np.mean(ll_short))
    if write:
        with open(path+'.short', 'w') as f:
            f.write('\n'.join(lines_short))

def top_ngram(fld, in_fname, n, min_freq=20, max_n=10000):
    path = fld + '/' + in_fname
    print('finding %igram from %s'%(n, path))
    counter = Counter()
    for line in open(path, encoding='utf-8'):
        ww = [SOS_token] + line.strip('\n').split() + [EOS_token]
        for i in range(n, len(ww) + 1):
            ngram = ' '.join(ww[i - n: i])
            counter[ngram] += 1
    
    candidates = counter.most_common(max_n)
    final = []
    for ngram, count in candidates:
        if count < min_freq:
            break
        final.append(ngram)
    
    with open(fld + '/' + in_fname + '.%igram'%n, 'w', encoding='utf-8') as f:
        f.write('\n'.join(final))


def extract_tgt(path, n_max=1e6):
    lines = []
    for i, line in enumerate(open(path, encoding='utf-8')):
        _, tgt = line.strip('\n').split('\t')
        lines.append(tgt)
        if i == n_max:
            break
    with open(path+'.tgt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def build_dataset(fld, prob_T=0.1):
    shuffle_file(fld + '/negative.txt')
    shuffle_file(fld + '/positive.txt')
    mix_shuffle(fld + '/positive.txt.shuffled', fld+'/negative.txt.shuffled', fld, prob_T=prob_T)
    for name in ['vali', 'test', 'train']:
        txt2num(fld+'/' + name, fld+'/vocab.txt')