from util import SOS_token, EOS_token
import numpy as np
from sklearn import linear_model
from sklearn import metrics 

class ClassifierNgram:

    def __init__(self, fld, ngram):
        self.fld = fld
        self.ngram2ix = dict()
        self.ngram = ngram
        path_ngram = fld + '/%igram.txt'%ngram
        for i, line in enumerate(open(path_ngram, encoding='utf-8')):
            ngram = line.strip('\n')
            self.ngram2ix[ngram] = i
            assert(self.ngram == len(ngram.split()))
        self.vocab_size = i + 1
        print('loaded %i %igram'%(self.vocab_size, self.ngram))
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
            for i in range(self.ngram, len(ww) + 1):
                ngram = ' '.join(ww[i - self.ngram: i])
                ix = self.ngram2ix.get(ngram, None)
                if ix is not None:
                    X[m, ix] = 1.
            m += 1
            if m == batch:
                break
        return X[:m, :], np.array(y[:m])

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
                with open(fld + '/%igram_coef.txt'%self.ngram, 'w') as f:
                    f.write('\n'.join(['%.4f'%c for c in coef]))
                with open(fld + '/%igram_acc.txt'%self.ngram, 'w') as f:
                    f.write(str(acc))

    def load(self):
        coef = [float(line.strip('\n')) for line in open(self.fld + '/%igram_coef.txt'%self.ngram)]
        self.model.coef_ = np.reshape(coef, (1, -1))



                
