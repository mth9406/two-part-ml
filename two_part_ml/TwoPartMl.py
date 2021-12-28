import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error


class TwoPartsML(object):

    def __init__(self, clf, regr):
        self.clf, self.regr = clf, regr
        # attributes
        self.train_scores = dict()
        self.fitted = False

    def fit(self, 
            X_train, y_train,
            freq = 'FREQ', severity = 'EXPENDOP', 
            ):
        # fit self.clf and self.regr
        idx_tr = y_train[severity] > 0 

        # split dataset
        self.regr.fit(X_train.loc[idx_tr, :], y_train.loc[idx_tr, severity]) # regression
        self.clf.fit(X_train, y_train[freq]) # classifier
        self.fitted = True
        
        # save scores for the training dataset
        self.train_scores['regr_score'] = self.regr.score(X_train.loc[idx_tr, :], y_train.loc[idx_tr, severity])
        self.train_scores['clf_score'] = self.clf.score(X_train, y_train[freq])
        preds_tr = self.predict(X_train)
        self.train_scores['twoparts_score'] = r2_score(y_train[severity], preds_tr)
        

    def predict(self, X):
        # return predictions
        return self.clf.predict_proba(X)[:, 1]*self.regr.predict(X)

    def score(self, X, y):
        return r2_score(y, self.predict(X))

    @property
    def train_score(self):
        return self.train_scores

    def cal_test_score(self, X_test, y_test, freq= 'FREQ', severity= 'EXPENDOP'):
        test_scores = {'clf_score':self.clf.score(X_test, y_test[freq]),
                            'regr_score':self.regr.score(X_test, y_test[severity]),
                            'twoparts_score':self.score(X_test, y_test[severity])}
        return test_scores

    def mape(self, X, y):
        return mean_absolute_percentage_error(y,self.predict(X))

    def mae(self, X, y):
        return mean_absolute_error(y, self.predict(X))

    def cal_test_mape(self, X_test, y_test, freq= 'FREQ', severity= 'EXPENDOP'):
        test_scores = {'clf_score':self.clf.score(X_test, y_test[freq]),
                            'regr_score':mean_absolute_percentage_error(y_test[severity],self.regr.predict(X_test)),
                            'twoparts_score':self.mape(X_test, y_test[severity])}
        return test_scores

    def cal_test_mae(self, X_test, y_test, freq= 'FREQ', severity= 'EXPENDOP'):
        test_scores = {'clf_score':self.clf.score(X_test, y_test[freq]),
                            'regr_score':mean_absolute_error(y_test[severity],self.regr.predict(X_test)),
                            'twoparts_score':self.mae(X_test, y_test[severity])}
        return test_scores
