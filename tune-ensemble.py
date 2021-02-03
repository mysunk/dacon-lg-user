"""
Created on Mon Mar  2 22:44:55 2020

@author: guseh
"""
# packages
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, STATUS_FAIL
from functools import partial
import pandas as pd
# models
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from sklearn.metrics import *


N_FOLD = 10

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pickle
def save_obj(obj, name):
    with open('tune_results/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('tune_results/' + name + '.pkl', 'rb') as f:
        trials = pickle.load(f)
        trials = sorted(trials, key=lambda k: k['loss'])
        return trials


def make_param_int(param, key_names):
    for key, value in param.items():
        if key in key_names:
            param[key] = int(param[key])
    return param


def f_pr_auc(probas_pred, y_true):
    labels = y_true.get_label()
    p, r, _ = precision_recall_curve(labels, probas_pred)
    score = auc(r, p)
    return "pr_auc", score, True


from sklearn.model_selection import KFold
def lgb_train_model(train_x, train_y, params):
    '''
    cross validation with given data
    '''
    valid_probs = np.zeros((train_y.shape))
    # -------------------------------------------------------------------------------------
    # Kfold cross validation
    models = []
    k_fold = KFold(n_splits = N_FOLD, shuffle=True, random_state=0)
    for train_idx, val_idx in k_fold.split(train_x):
        # split train, validation set
        if type(train_x) == pd.DataFrame:
            X = train_x.iloc[train_idx,:]
            valid_x = train_x.iloc[val_idx, :]
        elif type(train_x) == np.ndarray:
            X = train_x[train_idx, :]
            valid_x = train_x[val_idx, :]
        else:
            print('Unknown data type for X')
            # return -1, -1
        y = train_y[train_idx]
        valid_y = train_y[val_idx]

        d_train = lgb.Dataset(X, y)
        d_val = lgb.Dataset(valid_x, valid_y)

        # run training
        model = lgb.train(
            params,
            train_set=d_train,
            num_boost_round=1000,
            valid_sets=d_val,
            early_stopping_rounds=100,
            feval=f_pr_auc,
            verbose_eval=False,
        )

        # cal valid prediction
        valid_prob = model.predict(valid_x)
        valid_probs[val_idx] = valid_prob

    # cv score
    auc_score = roc_auc_score(train_y, valid_probs)
    return auc_score

import catboost as cbm
def cat_train_model(train_x, train_y, params):
    '''
    cross validation with given data
    '''
    valid_probs = np.zeros((train_y.shape))
    # -------------------------------------------------------------------------------------
    # Kfold cross validation
    k_fold = KFold(n_splits=N_FOLD, shuffle=True, random_state=0)
    for train_idx, val_idx in k_fold.split(train_x):
        # split train, validation set
        if type(train_x) == pd.DataFrame:
            X = train_x.iloc[train_idx, :]
            valid_x = train_x.iloc[val_idx, :]
        elif type(train_x) == np.ndarray:
            X = train_x[train_idx, :]
            valid_x = train_x[val_idx, :]
        else:
            print('Unknown data type for X')
            # return -1, -1
        y = train_y[train_idx]
        valid_y = train_y[val_idx]

        from catboost import CatBoostClassifier, Pool
        train_dataset = Pool(data=X,
                     label=y,
                     cat_features=['model_start','model_end'])

        valid_dataset = Pool(data=valid_x,
                     label=valid_y,
                     cat_features=['model_start','model_end'])

        cbm_clf = CatBoostClassifier(**params)

        cbm_clf.fit(train_dataset,
            eval_set=valid_dataset,
            verbose=False,
            plot=False,
        )

        # cal valid prediction
        valid_prob = cbm_clf.predict_proba(valid_x)
        valid_probs[val_idx] = valid_prob[:,1]

    # cv score
    auc_score = roc_auc_score(train_y, valid_probs)
    return auc_score

from sklearn.ensemble import ExtraTreesClassifier
def ext_train_model(train_x, train_y, params):
    valid_probs = np.zeros((train_y.shape))
    # -------------------------------------------------------------------------------------
    # Kfold cross validation
    k_fold = KFold(n_splits=N_FOLD, shuffle=True, random_state=0)
    for train_idx, val_idx in k_fold.split(train_x):
        # split train, validation set
        if type(train_x) == pd.DataFrame:
            X = train_x.iloc[train_idx, :]
            valid_x = train_x.iloc[val_idx, :]
        elif type(train_x) == np.ndarray:
            X = train_x[train_idx, :]
            valid_x = train_x[val_idx, :]
        else:
            print('Unknown data type for X')
            # return -1, -1
        y = train_y[train_idx]
        valid_y = train_y[val_idx]

        model = ExtraTreesClassifier(**params)
        model.fit(X, y)

        # cal valid prediction
        valid_prob = model.predict_proba(valid_x)
        valid_probs[val_idx] = valid_prob[:,1]

    # cv score
    auc_score = roc_auc_score(train_y, valid_probs)
    return auc_score


class Tuning_model(object):
    def __init__(self):
        self.random_state = 0
        self.space = {}

    def extra_space(self):
        self.space = {
            'max_depth':                hp.quniform('max_depth', 20, 40, 1),
            'n_estimators':             hp.quniform('n_estimators', 1000, 3000, 50),
            'min_samples_leaf':         hp.quniform('min_samples_leaf', 1, 30, 1),
            'min_samples_split':        hp.uniform('min_samples_split', 0, 0.1),
            'criterion':                hp.choice('criterion', ['gini', 'entropy']),
            'n_jobs':                   -1
            }

    def lgb_space(self):
        # LightGBM parameters
        self.space = {
            'objective':                'binary',
            # 'min_child_weight':         hp.quniform('min_child_weight', 5, 15, 1),
            'learning_rate':            hp.uniform('learning_rate',    0.015, 0.018),
            'max_depth':                -1,
            'num_leaves':               hp.quniform('num_leaves',       35, 45, 1),
            'min_data_in_leaf':		    hp.quniform('min_data_in_leaf',	25, 30, 1),	# overfitting 안되려면 높은 값
            'reg_alpha':                hp.uniform('reg_alpha',0.85, 0.95),
            'reg_lambda':               hp.uniform('reg_lambda',0.009,0.013),
            'colsample_bytree':         hp.uniform('colsample_bytree', 0.15, 0.25),
            'colsample_bynode':		    hp.uniform('colsample_bynode',0.35,0.5),
            'bagging_freq':			    hp.quniform('bagging_freq',	10,25,1),
            'tree_learner':			    hp.choice('tree_learner',	['feature','data','voting']),
            'subsample':                hp.uniform('subsample', 0.99, 1.0),
            'boosting':			        hp.choice('boosting', ['gbdt']),
            'max_bin':			        hp.quniform('max_bin',		2,7,1), # overfitting 안되려면 낮은 값
            "min_sum_hessian_in_leaf":  hp.uniform('min_sum_hessian_in_leaf',       0.05,0.1),
            'random_state':             self.random_state,
            'n_jobs':                   -1,
            'metrics':                  'auc',
            'verbose':                  -1,
            'force_col_wise':           True,
        }

    def cat_space(self):
        self.space = {
            'custom_loss':              'AUC',
            'learning_rate':            hp.uniform('learning_rate', 0.001, 0.1),
            'bagging_temperature':      hp.uniform('bagging_temperature', 0, 1.0),
            # 'colsample_bylevel':        hp.uniform('colsample_bylevel', 0, 1),
            'subsample':                hp.uniform('subsample',       0, 1),
            'scale_pos_weight':         hp.uniform('scale_pos_weight', 0.01, 1.0),
            'random_strength':          hp.loguniform('random_strength', 1e-9, 10),

            'border_count':             hp.quniform('border_count', 1, 255, 1),
            'depth':                    hp.quniform('depth', 1, 10, 1),
            'l2_leaf_reg':              hp.quniform('l2_leaf_reg', 2, 50, 1),
            'iterations':               10000,
            'task_type':'GPU',
            'use_best_model': True,
            'od_type' : 'Iter',
            'bootstrap_type' : 'Poisson',
            'devices' : '0:1'
            }

    # optimize
    def process(self, clf_name, train_set, trials, algo, max_evals):
        fn = getattr(self, clf_name+'_cv')
        space = getattr(self, clf_name+'_space')
        space()
        fmin_objective = partial(fn, train_set=train_set)
        try:
            result = fmin(fn=fmin_objective, space=self.space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials

    def cat_cv(self, params, train_set):
        params = make_param_int(params, ['border_count', 'depth','l2_leaf_reg', 'reg_lambda', 'iterations'])
        train_x, train_y = train_set
        best_loss = cat_train_model(train_x, train_y, params)
        return {'loss': -best_loss, 'params': params, 'status': STATUS_OK}

    def extra_cv(self, params, train_set):
        params = make_param_int(params, ['max_depth', 'min_samples_leaf', 'n_estimators'])
        train_x, train_y = train_set
        best_loss = ext_train_model(train_x, train_y, params)
        return {'loss': -best_loss, 'params': params, 'status': STATUS_OK}

    def lgb_cv(self, params, train_set):
        params = make_param_int(params, ['max_depth', 'num_leaves', 'min_data_in_leaf',
                                         'min_child_weight', 'bagging_freq', 'max_bin'])

        train_x, train_y = train_set
        best_loss = lgb_train_model(train_x, train_y, params)
        return {'loss': -best_loss, 'params': params, 'status': STATUS_OK}

if __name__ == '__main__':

    # load config
    parser = argparse.ArgumentParser(description='Tune each household...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--method', default='lgb', choices=['extra','cat','lgb'])
    parser.add_argument('--max_evals', default=1000, type=int)
    parser.add_argument('--save_file', default='tmp')
    args = parser.parse_args()

    # load dataset
    train = pd.read_csv('train_X_fin.csv', index_col=0)
    train_label = np.load('train_y.npy')

    train['model_start'] = pd.Series(train['model_start'], dtype='category')
    train['model_end'] = pd.Series(train['model_end'], dtype='category')

    # main
    clf = args.method
    bayes_trials = Trials()
    obj = Tuning_model()
    tuning_algo = tpe.suggest # -- bayesian opt
    # tuning_algo = tpe.rand.suggest # -- random search
    obj.process(args.method, [train, train_label],
                           bayes_trials, tuning_algo, args.max_evals)

    # save trial
    save_obj(bayes_trials.results,args.save_file)