"""
Created on Mon Mar  2 22:44:55 2020

@author: guseh
"""
# packages
import argparse
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, STATUS_FAIL
from functools import partial
from util import *

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


class Tuning_model(object):

    def __init__(self):
        self.random_state = 0
        self.space = {}

    # parameter setting
    def lgb_space(self):
        # LightGBM parameters
        self.space = {
            'objective':                'binary',
            'learning_rate':            hp.uniform('learning_rate',    0.001, 0.1),
            'max_depth':                -1,
            'num_leaves':               hp.quniform('num_leaves',       5, 300, 1),
            'min_data_in_leaf':		    hp.quniform('min_data_in_leaf',	50, 300, 1),	# overfitting 안되려면 높은 값
            'reg_alpha':                hp.uniform('reg_alpha',0,1),
            'reg_lambda':               hp.uniform('reg_lambda',0, 1),
            'min_child_weight':         hp.quniform('min_child_weight', 1, 30, 1),
            'colsample_bytree':         hp.uniform('colsample_bytree', 0.01, 1.0),
            'colsample_bynode':		    hp.uniform('colsample_bynode',0.01,1.0),
            'bagging_freq':			    hp.quniform('bagging_freq',	1,20,1),
            'tree_learner':			    hp.choice('tree_learner',	['serial','feature','data','voting']),
            'subsample':                hp.uniform('subsample', 0.01, 1.0),
            'boosting':			        hp.choice('boosting', ['gbdt','rf']),
            'max_bin':			        hp.quniform('max_bin',		3,50,1), # overfitting 안되려면 낮은 값
            "min_sum_hessian_in_leaf": hp.quniform('min_sum_hessian_in_leaf',       5, 15, 1),
            'random_state':             self.random_state,
            'n_jobs':                   -1,
            'metrics':                  'auc',
            'verbose':                  -1,
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

    def lgb_cv(self, params, train_set):
        params = make_param_int(params, ['max_depth','num_leaves','min_data_in_leaf',
                                     'min_child_weight','bagging_freq','max_bin','min_sum_hessian_in_leaf'])

        models = []
        train_x, train_y = train_set
        valid_probs = np.zeros((train_y.shape))
        # -------------------------------------------------------------------------------------
        # 5 Kfold cross validation
        k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
        for train_idx, val_idx in k_fold.split(train_x):
            # split train, validation set
            X = train_x[train_idx]
            y = train_y[train_idx]
            valid_x = train_x[val_idx]
            valid_y = train_y[val_idx]

            d_train = lgb.Dataset(X, y)
            d_val = lgb.Dataset(valid_x, valid_y)

            # run training
            model = lgb.train(
                params,
                train_set=d_train,
                num_boost_round=1000,
                valid_sets=d_val,
                feval=f_pr_auc,
                early_stopping_rounds=10,
                verbose_eval=False
            )

            # cal valid prediction
            valid_prob = model.predict(valid_x)
            valid_probs[val_idx] = valid_prob

            models.append(model)
        best_loss = roc_auc_score(train_y, valid_probs)
        # Dictionary with information for evaluation
        return {'loss': -best_loss, 'params': params, 'status': STATUS_OK}

if __name__ == '__main__':

    # load config
    parser = argparse.ArgumentParser(description='Tune each household...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--method', default='lgb', choices=['lgb'])
    parser.add_argument('--max_evals', default=1000, type=int)
    parser.add_argument('--save_file', default='0118')
    args = parser.parse_args()

    # load_dataset
    data_path = 'data/'
    train_err_arr = np.load(f'{data_path}/train_err_arr.npy')
    train_problem_arr = np.load(f'{data_path}/train_problem_arr.npy')

    WINDOW = 3
    train_err_list = []
    for i in range(31 - WINDOW):
        sum_ = np.sum(train_err_arr[:, i:i + WINDOW, :], axis=1)
        train_err_list.append(sum_)
    train_err_r = np.concatenate(
        [np.min(train_err_list, axis=0), np.max(train_err_list, axis=0), np.mean(train_err_list, axis=0)], axis=1)
    # y
    train_problem_r = np.max(train_problem_arr, axis=1)

    # model train
    train_y = (train_problem_r > 0).astype(int)

    # main
    clf = args.method
    bayes_trials = Trials()
    obj = Tuning_model()
    tuning_algo = tpe.suggest # -- bayesian opt
    # tuning_algo = tpe.rand.suggest # -- random search
    obj.process(args.method, [train_err_r, train_y],
                           bayes_trials, tuning_algo, args.max_evals)

    # save trial
    save_obj(bayes_trials.results,args.save_file)