from util import *
from sklearn import linear_model
from sklearn import metrics 
import pickle

class ClassifierNgram:

    def __init__(self, fld, ngram, include_punc=False, model_class='logistic'):
        self.fld = fld
        self.ngram2ix = dict()
        self.ngram = ngram
        self.include_punc = include_punc

        fname = '%igram'%ngram
        if include_punc:
            fname +=  '.include_punc'
        self.path_prefix = fld + '/' + fname
        for i, line in enumerate(open(self.path_prefix + '.txt', encoding='utf-8')):
            ngram = line.strip('\n')
            self.ngram2ix[ngram] = i
            assert(self.ngram == len(ngram.split()))
        self.vocab_size = i + 1
        print('loaded %i %igram'%(self.vocab_size, self.ngram))
        self.model_class = model_class
        if self.model_class == 'logistic':
            self.model = linear_model.SGDClassifier(loss='logistic', random_state=9, max_iter=1, tol=1e-3)
        elif self.model_class == 'linear':
            self.model = linear_model.LinearRegression()
        


    def get_Xy(self, f, batch):
        txts = []
        y = []
        m = 0
        for line in f:
            x_, y_ = line.strip('\n').split('\t')
            txts.append(x_)
            y.append(float(y_))
            m += 1
            if m == batch:
                break
        return self.txts2mat(txts), np.array(y[:m])


    def txts2mat(self, txts):
        X = np.zeros((len(txts), self.vocab_size))
        for i, txt in enumerate(txts):
            ww = txt2ww(txt, self.include_punc)
            for t in range(self.ngram, len(ww) + 1):
                ngram = ' '.join(ww[t - self.ngram: t])
                j = self.ngram2ix.get(ngram, None)
                if j is not None:
                    X[i, j] = 1.
        return X


    def fit(self, batch):
        X_vali, y_vali = self.get_Xy(open(self.fld + '/vali.txt', encoding='utf-8'), 1000)
        f_train = open(self.fld + '/train.txt', encoding='utf-8')
        n_trained = 0
        max_vali_acc = 0.
        while True:
            X, y = self.get_Xy(f_train, batch)
            if X.shape[0] == 0:
                break
            print('fitting...')
            if self.model_class == 'logistic':
                self.model = self.model.partial_fit(X, y, classes=np.array([0,1]))
            else:
                self.model = self.model.fit(X, y)

            n_trained += len(y)

            if self.model_class == 'linear':
                y_pred = self.model.predict(X_vali)
                acc = sum([int(y_pred[i]>0.5) == int(y_vali[i]>0.5) for i in range(len(y_vali))])/len(y_vali)
            else:
                y_pred = self.model.predict_proba(X_vali)
                acc = sum([int(y_pred[i,1]>0.5) == y_vali[i] for i in range(len(y_vali))])/len(y_vali)
                log_loss = metrics.log_loss(y_vali, y_pred)
                print('trained %.1f k, log_loss = %.3f, acc = %3f'%(n_trained/1000, log_loss, acc))

            if acc > max_vali_acc:
                max_vali_acc = acc
                print('saving best coef')
                coef = self.model.coef_.ravel()
                with open(self.path_prefix + '.coef', 'w') as f:
                    f.write('\n'.join(['%.4f'%c for c in coef]))
                with open(self.path_prefix + '.acc', 'w') as f:
                    f.write(str(acc))
                pickle.dump(self.model, open(self.path_prefix + '.p', 'wb'))

    def load(self):
        self.model = pickle.load(open(self.path_prefix + '.p', 'rb'))

    def predict(self, txts):
        data = self.txts2mat(txts)
        prob = self.model.predict_proba(data)
        return prob[:,1]




class ClassifierNgramEnsemble:
                
    def __init__(self, fld, include_punc=False):
        self.fld = fld
        self.children = dict()
        self.wt = dict()
        for ngram in [1, 2, 3, 4]:
            self.children[ngram] = ClassifierNgram(fld, ngram, include_punc)
            self.children[ngram].load()
            acc = float(open(self.children[ngram].path_prefix + '.acc').readline().strip('\n'))
            self.wt[ngram] = 2. * max(0, acc - 0.5)

    def predict(self, txts):
        avg_scores = np.array([0.] * len(txts))
        for ngram in self.children:
            scores = self.children[ngram].predict(txts)
            avg_scores += scores * self.wt[ngram]
        return avg_scores / sum(self.wt.values())

    def test(self):
        txts = []
        y_vali = []
        m = 0
        for line in open(self.fld + '/test.txt', encoding='utf-8'):
            txt, y = line.strip('\n').split('\t')
        #for line in open(self.fld + '/positive.txt', encoding='utf-8'):
        #    txt = line.strip('\n'); y = 1.
            txts.append(txt)
            y_vali.append(float(y))
            m += 1
            if m == 1000:
                break

        for ngram in self.children:
            score = self.children[ngram].predict(txts)
            acc = sum([int(score[i]>0.5) == y_vali[i] for i in range(len(y_vali))])/len(y_vali)
            print('%igram acc = %.4f'%(ngram, acc))

        score = self.predict(txts)
        acc = sum([int(score[i]>0.5) == y_vali[i] for i in range(len(y_vali))])/len(y_vali)
        print('ensemble acc = %.4f'%acc)



class Classifier1gramCount:
    def __init__(self, fld):
        self.fld = fld

    def fit(self, min_freq=60, max_n=1e5):
        scores = dict()
        n = 0
        for line in open(self.fld + '/all.txt', encoding='utf-8'):
            n += 1
            cells = line.strip('\n').split('\t')
            if len(cells) != 2:
                print(cells)
                exit()
            txt, score = cells
            for w in set(txt.strip().split()):
                if is_word(w):
                    if w not in scores:
                        scores[w] = []
                    scores[w].append(float(score))
            if n == max_n:
                break


        lines = ['\t'.join(['word', 'avg', 'se', 'count'])]
        for w in scores:
            count = len(scores[w])
            if count < min_freq:
                continue
            avg = np.mean(scores[w])
            se = np.std(scores[w])/np.sqrt(count)
            lines.append('\t'.join([w, '%.4f'%avg, '%.4f'%se, '%i'%count]))

        with open(self.fld + '/count.tsv', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def load(self):
        self.coef = dict()
        f = open(self.fld + '/count.tsv', encoding='utf-8')
        header = f.readline()
        for line in f:
            w, avg = line.strip('\n').split('\t')[:2]
            self.coef[w] = float(avg)

    def corpus_score(self, txts, kw=100):
        scores = []
        coef_w = []
        for w in self.coef:
            coef_w.append((self.coef[w], w))
        coef_w = sorted(coef_w, reverse=True)[:kw]
        print('last:',coef_w[-1])
        keywords = set([w for _, w in coef_w])

        #total_joint = 0
        #total = 0

        for txt in txts:
            words = set()
            for w in txt.strip().split():
                if is_word(w):
                    words.add(w)
            joint = words & keywords
            scores.append(len(joint)/len(words))
            #total_joint += len(joint)
            #total += len(words)
        return np.mean(scores), np.std(scores)/np.sqrt(len(scores))
        #return total_joint/total


    def test(self, kw=100):
        import matplotlib.pyplot as plt

        txts = []
        labels = []
        for line in open(self.fld + '/sorted_avg.tsv', encoding='utf-8'):
            txt, label = line.strip('\n').split('\t')
            txts.append(txt)
            labels.append(float(label))

        i0 = 0
        human = []
        pred = []
        while True:
            i1 = i0 + 100
            if i1 >= len(txts):
                break
            human.append(np.mean(labels[i0:i1]))
            pred.append(self.corpus_score(txts[i0:i1], kw=kw))
            i0 = i1

        plt.plot(human, pred, '.')
        plt.xlabel('human')
        plt.xlabel('metric (ratio of keywords)')
        plt.title('corr = %.4f'%np.corrcoef(human, pred)[0][1])
        plt.savefig(self.fld + '/test_corr_kw%i.png'%kw)

                    


