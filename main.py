#%% import
import pandas as pd
import numpy as np
import matplotlib
from tqdm import tqdm
font = {'size': 16, 'family':"NanumGothic"}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

data_path = 'data/'
NUM_ERR_TYPES = 42

#%% train data load
train_err = pd.read_csv(data_path + 'train_err_data.csv')
# train_quality = pd.read_csv(data_path + 'train_quality_data_cleansed.csv')
train_problem = pd.read_csv(data_path + 'train_problem_data.csv')
train_err['time'] = pd.to_datetime(train_err['time'], format='%Y%m%d%H%M%S')

print('Train data load done')

#%% train-val split
train_user_id = np.unique(train_err['user_id'])
TRAIN_SPLIT = int(len(train_user_id) * 0.8)

#%% train, val preprocessing
def preprocessing(user_ids, err, problem):
    train_problem_count = dict()
    train_err_count = dict()
    for user_id in tqdm(user_ids):

        # find data corresponding to user_id
        idx_err = err['user_id'] == user_id
        # idx_quality = train_quality['user_id'] == user_id
        idx_problem = problem['user_id'] == user_id

        # strip data
        train_err_u = err.loc[idx_err, :]
        # train_quality_u = train_quality.loc[idx_quality, :]
        train_problem_u = problem.loc[idx_problem, 'time']

        # make label
        train_problem_count[user_id] = train_problem_u.shape[0]

        # cut train_err w.r.t w_b and w_a
        time_ranges = []
        for i in range(train_problem_u.shape[0]):
            prob_time = train_problem_u.iloc[i]
            # error 범위 한정
            idx_interest_b = \
                (train_err_u['time'] < pd.to_datetime(prob_time, format='%Y%m%d%H%M%S')).values & \
                (train_err_u['time'] > pd.to_datetime(prob_time, format='%Y%m%d%H%M%S') - pd.Timedelta(w_b,
                                                                                                       unit='h')).values
            idx_interest_a = \
                (train_err_u['time'] > pd.to_datetime(prob_time, format='%Y%m%d%H%M%S')).values & \
                (train_err_u['time'] < pd.to_datetime(prob_time, format='%Y%m%d%H%M%S') + pd.Timedelta(w_a,
                                                                                                       unit='h')).values
            time_range = idx_interest_b | idx_interest_a
            time_ranges.append(time_range)
        time_ranges = np.array(time_ranges).T
        if len(time_ranges) == 0:
            err_count = np.zeros((42), dtype=int)
            train_err_count[user_id] = err_count
            continue
        time_ranges = np.any(time_ranges, axis=1)
        train_err_u = train_err_u.iloc[time_ranges, -2:]

        err_count = np.zeros((42), dtype=int)
        for i in range(1, NUM_ERR_TYPES + 1):
            # errcode 미사용
            err_count[i - 1] = (train_err_u['errtype'] == i).sum()
        train_err_count[user_id] = err_count
    train_err_count = pd.DataFrame(train_err_count).T
    train_problem_count = pd.DataFrame(train_problem_count,index = [0]).T
    return train_err_count, train_problem_count

# window length [hour]
w_b = 24*3
w_a = 24*6

train_x, train_y = preprocessing(train_user_id, train_err, train_problem)
train_y[train_y > 0] = 1

print('Preprocessing done')

del train_err, train_problem

#%% model train:: baseline
from sklearn.metrics import *
from sklearn.model_selection import KFold
import lightgbm as lgb

# validation auc score를 확인하기 위해 정의
def f_pr_auc(probas_pred, y_true):
    labels = y_true.get_label()
    p, r, _ = precision_recall_curve(labels, probas_pred)
    score = auc(r, p)
    return "pr_auc", score, True

models = []
recalls = []
precisions = []
auc_scores = []
threshold = 0.5
# 파라미터 설정
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'seed': 1015
}
# -------------------------------------------------------------------------------------
# 5 Kfold cross validation
k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
for train_idx, val_idx in k_fold.split(train_x):
    # split train, validation set
    X = train_x.iloc[train_idx,:]
    y = train_y.iloc[train_idx,:]
    valid_x = train_x.iloc[val_idx,:]
    valid_y = train_y.iloc[val_idx,:]

    d_train = lgb.Dataset(X, y)
    d_val = lgb.Dataset(valid_x, valid_y)

    # run traning
    model = lgb.train(
        params,
        train_set=d_train,
        num_boost_round=1000,
        valid_sets=d_val,
        feval=f_pr_auc,
        verbose_eval=20,
        early_stopping_rounds=3
    )

    # cal valid prediction
    valid_prob = model.predict(valid_x)
    valid_pred = np.where(valid_prob > threshold, 1, 0)

    # cal scores
    recall = recall_score(valid_y, valid_pred)
    precision = precision_score(valid_y, valid_pred)
    auc_score = roc_auc_score(valid_y, valid_prob)

    # append scores
    models.append(model)
    recalls.append(recall)
    precisions.append(precision)
    auc_scores.append(auc_score)

    print('==========================================================')
    print(np.where(model.feature_importance())[0] + 1)

#%% 교차검증 점수 확인
print(np.mean(auc_scores))

#%% feature importance
plt.plot()
plt.show()