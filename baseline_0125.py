#%% 라이브러리 임포트 및 함수 로드
import matplotlib
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import *
import pandas as pd

font = {'size': 16, 'family':"NanumGothic"}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

data_path = 'data/'

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
        X = train_x[train_idx,:]
        y = train_y[train_idx]
        valid_x = train_x[val_idx,:]
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
            early_stopping_rounds=100,
            verbose_eval=True,
        )

        # cal valid prediction
        valid_prob = model.predict(valid_x)
        valid_probs[val_idx] = valid_prob

        models.append(model)

    return models, valid_probs


#%% load dataset
train_err_arr = np.load(f'{data_path}/train_err_arr.npy')
train_problem_arr = np.load(f'{data_path}/train_problem_arr.npy')
test_err_arr = np.load(f'{data_path}/test_err_arr.npy')


#%% evaluate
## train
# X
idx = np.ones((42), dtype=bool)
idx_drop = np.array([ 1,  7,  8, 19, 20, 28])
idx[idx_drop] = False

WINDOW = 1
train_err_list = []
for i in range(31-WINDOW):
    sum_ = np.sum(train_err_arr[:,i:i+WINDOW,idx], axis=1)
    train_err_list.append(sum_)
train_err_r = np.concatenate([np.min(train_err_list, axis=0), np.max(train_err_list, axis=0), np.mean(train_err_list, axis=0),
                              np.median(train_err_list, axis=0)], axis=1)

# y
train_problem_r = np.max(train_problem_arr, axis=1)
train_y = (train_problem_r > 0).astype(int)


params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'seed': 1015,
    'verbose': 0,
}

# 모델 학습
models, valid_probs = train_model(train_err_r, train_y, params)

# evaluate
threshold = 0.5
valid_preds = np.where(valid_probs > threshold, 1, 0)

# cal scores
recall = recall_score(train_y, valid_preds)
precision = precision_score(train_y, valid_preds)
auc_score = roc_auc_score(train_y, valid_probs)
print(auc_score)

#%% 제출
submission = pd.read_csv(data_path + 'sample_submission.csv')

test_err_list = []
for i in range(31-WINDOW):
    sum_ = np.sum(test_err_arr[:,i:i+WINDOW,idx], axis=1)
    test_err_list.append(sum_)
test_err_r = np.concatenate([np.min(test_err_list, axis=0), np.max(test_err_list, axis=0), np.mean(test_err_list, axis=0),
                             np.median(test_err_list, axis=0)], axis=1)

# predict
test_prob = []
for model in models:
    test_prob.append(model.predict(test_err_r))
test_prob = np.mean(test_prob, axis=0)

submission['problem'] = test_prob.reshape(-1)
submission.to_csv("submission.csv", index = False)