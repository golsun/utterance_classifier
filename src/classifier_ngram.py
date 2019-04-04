from util import SOS_token, EOS_token
import numpy as np
from sklearn import linear_model
from sklearn import metrics 

class ClassifierNgram:

    def __init__(self, path_ngram):
        self.ngrams = dict()
        self.n = None
        for i, line in enumerate(open(path_ngram, encoding='utf-8')):
            ngram = line.strip('\n')
            self.ngrams[ngram] = i
            if self.n is None:
                self.n = len(ngram.split())
            else:
                assert(self.n == len(ngram.split()))
        self.vocab_size = i + 1
        print('loaded %i %igram'%(self.vocab_size, self.n))
        #self.model = LogisticRegression(solver='sag')#, max_iter=10)
        self.model = linear_model.SGDClassifier(loss='log', random_state=9, max_iter=1, tol=1e-3)

    def get_Xy(self, f, batch):
        X = np.zeros((batch, self.vocab_size))
        y = []
        m = 0
        for line in f:
            x_, y_ = line.strip('\n').split('\t')
            y.append(float(y_))
            ww = [SOS_token] + x_.split() + [EOS_token]
            for i in range(self.n, len(ww) + 1):
                ngram = ' '.join(ww[i - self.n: i])
                ix = self.ngrams.get(ngram, None)
                if ix is not None:
                    X[m, ix] = 1.
            m += 1
            if m == batch:
                break
        return X[:m, :], np.array(y[:m])

    def fit(self, fld, batch):
        X_vali, y_vali = self.get_Xy(open(fld + '/vali.txt', encoding='utf-8'), 1000)
        f_train = open(fld + '/train.txt', encoding='utf-8')
        n_trained = 0
        max_vali_acc = 0.
        while True:
            X, y = self.get_Xy(f_train, batch)
            if X.shape[0] == 0:
                break
            print('fitting...')
            self.model = self.model.partial_fit(X, y, classes=np.array([0,1]))
            n_trained += len(y)

            y_pred = self.model.predict_proba(X_vali)
            acc = sum([int(y_pred[i,1]>0.5) == y_vali[i] for i in range(len(y_vali))])/len(y_vali)
            log_loss = metrics.log_loss(y_vali, y_pred)
            print('trained %.1f k, log_loss = %.3f, acc = %3f'%(n_trained/1000, log_loss, acc))

            if acc > max_vali_acc:
                max_vali_acc = acc
                print('saving best coef')
                coef = self.model.coef_.ravel()
                with open(fld + '/%igram_coef.txt'%self.n, 'w') as f:
                    f.write('\n'.join(['%.4f'%c for c in coef]))

                
