from sklearn.metrics import *
from sklearn.model_selection import KFold
import lightgbm as lgb
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
import os

NUM_ERR_TYPES = 42
threshold = 0.5

def save_obj(obj, name):
    try:
        with open('tune_results/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    except FileNotFoundError:
        os.mkdir('tune_results')
        with open('tune_results/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('tune_results/' + name + '.pkl', 'rb') as f:
        trials = pickle.load(f)
        trials = sorted(trials, key=lambda k: k['loss'])
        return trials


def process_err(err_df, user_id, WINDOW = 24):
    '''
    error data를 일별 누적값으로 바꿈
    '''
    ## 1. user_id에 해당하는 data만 남김
    idx_err = err_df['user_id'] == user_id
    err_df = err_df.loc[idx_err, :]

    ## 2. WINDOW에 해당하는 날만 남김
    start_datetime = pd.to_datetime('2020-11-1')
    err_count_list = []
    while start_datetime <= pd.to_datetime('2020-12-1') - pd.Timedelta(WINDOW, unit='h'):
        end_datetime = start_datetime + pd.Timedelta(WINDOW, unit='h')  # 하루씩 lag
        err_count = np.zeros((42), dtype=int)
        if err_df.size == 0:
            pass
        else:
            idx_train_err = (err_df['time'] >= start_datetime).values & \
                            (err_df['time'] < end_datetime).values
            for i in range(1, NUM_ERR_TYPES + 1):
                # errcode 미사용
                err_count[i - 1] = (err_df['errtype'][idx_train_err] == i).sum()

        err_count_list.append(err_count)

        # update
        start_datetime += pd.Timedelta(1, unit='day')

    # Dimension reduction
    err_count_list = np.array(err_count_list)

    return err_count_list


def process_prob(prob_df, user_id, WINDOW = 24):
    '''
    problem을 일별로 바꿈
    '''
    ## 1. user_id에 해당하는 data만 남김
    idx_problem = prob_df['user_id'] == user_id
    prob_df = prob_df.loc[idx_problem, :]

    ## 2. WINDOW에 해당하는 날만 남김
    start_datetime = pd.to_datetime('2020-11-1')
    prob_count_list = []
    while start_datetime <= pd.to_datetime('2020-12-1') - pd.Timedelta(WINDOW, unit='h'):
        end_datetime = start_datetime + pd.Timedelta(WINDOW, unit='h')  # 하루씩 lag
        if len(prob_df) == 0:
            prob_count = 0
        else:
            idx_problem = (prob_df['time'] >= start_datetime).values & \
                          (prob_df['time'] < end_datetime).values
            prob_count = idx_problem.sum()

        prob_count_list.append(prob_count)

        # update
        start_datetime += pd.Timedelta(1, unit='day')

    # Dimension reduction
    prob_count_list = np.array(prob_count_list)

    return prob_count_list


def process_train(err_df, problem_df, train_user_id, WINDOW):
    err_arr = []
    problem_arr = []
    for user_id in tqdm(train_user_id):
        err = process_err(err_df, user_id, WINDOW)
        prob = process_prob(problem_df, user_id, WINDOW)
        err_arr.append(err)
        problem_arr.append(prob)

    err_arr = np.concatenate(err_arr, axis=0)
    problem_arr = np.concatenate(problem_arr, axis=0)

    return err_arr, problem_arr


def process_test(err_df, test_user_id):
    err_arr = []
    for user_id in tqdm(test_user_id):
        err = process_err(err_df, user_id)
        err_arr.append(err)

    err_arr = np.concatenate(err_arr, axis=0)

    return err_arr


def f_pr_auc(probas_pred, y_true):
    labels = y_true.get_label()
    p, r, _ = precision_recall_curve(labels, probas_pred)
    score = auc(r, p)
    return "pr_auc", score, True


def train_model(train_x, train_y, params):
    '''
    cross validation with given data
    '''
    models = []
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
            verbose_eval=True
        )

        # cal valid prediction
        valid_prob = model.predict(valid_x)
        valid_probs[val_idx] = valid_prob

        models.append(model)

    return models, valid_probs


def evaluate(valid_prob, valid_y):

    valid_pred = np.where(valid_prob > threshold, 1, 0)

    # cal scores
    recall = recall_score(valid_y, valid_pred)
    precision = precision_score(valid_y, valid_pred)
    auc_score = roc_auc_score(valid_y, valid_prob)

    print('==========================================================')
    # print(np.where(model.feature_importance())[0] + 1)

    return recall, precision, auc_score