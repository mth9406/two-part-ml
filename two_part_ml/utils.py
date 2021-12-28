import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def return_mse(model, X, y):
    preds = model.predict(X)
    mses = []
    for j in range(y.shape[1]):
        mses.append(np.mean((preds[:,j]-y[:,j])**2))
    return mses
    
def return_rmse(model, X, y):
    preds = model.predict(X)
    rmses = []
    for j in range(y.shape[1]):
        rmses.append(np.sqrt(np.mean((preds[:,j]-y[:,j])**2)))
    return rmses

def return_mape(model, X, y):
    preds = model.predict(X)
    rmses = []
    eps = 1e-6
    for j in range(y.shape[1]):
        rmses.append(np.mean(np.abs(preds[:,j]-y[:,j])/(y[:,j]+eps)))
    return rmses


def two_parts_cv(model, X, y, cv = 10):
    kf = KFold(n_splits=cv)
    train_scores = pd.DataFrame({'clf_score': [],
    'regr_score': [],
    'twoparts_score': []})
    test_scores = pd.DataFrame({'clf_score': [],
    'regr_score': [],
    'twoparts_score': []})
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.loc[train_index, :], X.loc[test_index, :]
        y_train, y_test = y.loc[train_index, :], y.loc[test_index, :]
        model.fit(X_train, y_train)
        train_scores = train_scores.append(model.train_score, ignore_index = True)
        test_scores = test_scores.append(model.cal_test_score(X_test, y_test), ignore_index= True)    
    return train_scores, test_scores


def two_parts_cv_mape(model, X, y, cv = 10):
    kf = KFold(n_splits=cv)
    train_scores = pd.DataFrame({'clf_score': [],
    'regr_mape': [],
    'twoparts_mape': []})
    test_scores = pd.DataFrame({'clf_score': [],
    'regr_mape': [],
    'twoparts_mape': []})
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.loc[train_index, :], X.loc[test_index, :]
        y_train, y_test = y.loc[train_index, :], y.loc[test_index, :]
        model.fit(X_train, y_train)
        train_scores = train_scores.append(model.cal_test_mape(X_train,y_train), ignore_index = True)
        test_scores = test_scores.append(model.cal_test_mape(X_test, y_test), ignore_index= True)    
    return train_scores, test_scores


def two_parts_cv_mae(model, X, y, cv = 10):
    kf = KFold(n_splits=cv)
    train_scores = pd.DataFrame({'clf_score': [],
    'regr_score': [],
    'twoparts_score': []})
    test_scores = pd.DataFrame({'clf_score': [],
    'regr_score': [],
    'twoparts_score': []})
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.loc[train_index, :], X.loc[test_index, :]
        y_train, y_test = y.loc[train_index, :], y.loc[test_index, :]
        model.fit(X_train, y_train)
        train_scores = train_scores.append(model.cal_test_mae(X_train,y_train), ignore_index = True)
        test_scores = test_scores.append(model.cal_test_mae(X_test, y_test), ignore_index= True)    
    return train_scores, test_scores


def separte_analysis_zero_nonzero_exp(X_train, X_test, y_train, y_test, model, target= 'EXPENDOP'):

    def convolution(scores, score):
        for k, v in scores.items():
            scores[k].append(score[k])

    ret = pd.DataFrame()
    print('training the model...')
    model.fit(X_train, y_train)  
    print('training done...')
    print('processing the analysis...')  
    
    scores = {'clf_score': [],
                'regr_score': [],
                'twoparts_score': []}

    convolution(scores, model.cal_test_mae(X_train, y_train))

    idx= (y_train[target]== 0)
    convolution(scores, model.cal_test_mae(X_train.loc[idx,:], y_train.loc[idx,:])) 

    idx= (y_train[target] >  0)
    convolution(scores, model.cal_test_mae(X_train.loc[idx,:], y_train.loc[idx,:]))

    convolution(scores, model.cal_test_mae(X_test, y_test))

    idx= (y_test[target]== 0)
    convolution(scores, model.cal_test_mae(X_test.loc[idx,:], y_test.loc[idx,:]))

    idx= (y_test[target]> 0)
    convolution(scores, model.cal_test_mae(X_test.loc[idx,:], y_test.loc[idx,:]))

    ret = pd.DataFrame(scores)
    ret.index = ['tr_overall', 'tr_zeros','tr_nonzeros','te_overall','te_zeroes','te_nonzeros']
    return ret

