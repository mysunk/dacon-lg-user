#%% import
# import matplotlib
from util import *
import numpy as np
import pandas as pd

font = {'size': 16, 'family':"NanumGothic"}
# matplotlib.rc('font', **font)
# import matplotlib.pyplot as plt
# plt.rcParams['axes.unicode_minus'] = False
#
data_path = 'data'
result_path = 'result_2'

#%% load dataset
train_err_arr = np.load(f'{data_path}/train_err_arr.npy')
train_problem_arr = np.load(f'{data_path}/train_problem_arr.npy')
test_err_arr = np.load(f'{data_path}/test_err_arr.npy')

#%%
import pickle
with open(f'{result_path}/tf_train_1.pkl', 'rb') as f:
    tf_train_1 = pickle.load(f)
with open(f'{result_path}/tf_train_2.pkl', 'rb') as f:
    tf_train_2 = pickle.load(f)
with open(f'{result_path}/tf_train_3.pkl', 'rb') as f:
    tf_train_3 = pickle.load(f)
with open(f'{result_path}/tf_test_1.pkl', 'rb') as f:
    tf_test_1 = pickle.load(f)
with open(f'{result_path}/tf_test_2.pkl', 'rb') as f:
    tf_test_2 = pickle.load(f)
with open(f'{result_path}/tf_test_3.pkl', 'rb') as f:
    tf_test_3 = pickle.load(f)

tf_train = pd.concat([tf_train_1, tf_train_2, tf_train_3], axis=0).reset_index(drop=True)
tf_test = pd.concat([tf_test_1, tf_test_2, tf_test_3], axis=0).reset_index(drop=True)

nan_idx = np.any(pd.isnull(pd.concat([tf_train, tf_test], axis=0)), axis=0)
tmp = pd.concat([tf_train, tf_test], axis=0)

tf_train = tf_train.loc[:,~nan_idx]
tf_test = tmp.iloc[15000:].reset_index(drop=True)
del tmp
#%%
# params = {
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': 'auc',
#     'seed': 1015,
#     'verbose': 0,
# }

params = load_obj('0118-3')[0]['params']

train_problem_r = np.max(train_problem_arr, axis=1)
train_y = (train_problem_r > 0).astype(int)
models, valid_probs = train_model(tf_train.values, train_y, params)

# evaluate
threshold = 0.5
valid_preds = np.where(valid_probs > threshold, 1, 0)

# cal scores
recall = recall_score(train_y, valid_preds)
precision = precision_score(train_y, valid_preds)
auc_score = roc_auc_score(train_y, valid_probs)
print(auc_score)

#%% test
submission = pd.read_csv(data_path + '/sample_submission.csv')

# predict
test_prob = []
for model in models:
    test_prob.append(model.predict(tf_test.values))
test_prob = np.mean(test_prob, axis=0)

submission['problem'] = test_prob.reshape(-1)
submission.to_csv("submit/submit_4.csv", index = False)
