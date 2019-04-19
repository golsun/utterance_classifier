from util import *
from sklearn import linear_model
from sklearn import metrics 
import pickle

class ClassifierNgram:

    def __init__(self, fld, ngram, include_punc=False):
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
        #self.model = LogisticRegression(solver='sag')#, max_iter=10)
        self.model = linear_model.SGDClassifier(loss='log', random_state=9, max_iter=1, tol=1e-3)


    def get_Xy(self, f, batch):
        X = np.zeros((batch, self.vocab_size))
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





                
