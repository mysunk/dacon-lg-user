#%% import
import matplotlib
from util import *

font = {'size': 16, 'family':"NanumGothic"}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

data_path = 'data/'

#%% load dataset
train_err_arr = np.load(f'{data_path}/train_err_arr.npy')
train_problem_arr = np.load(f'{data_path}/train_problem_arr.npy')
test_err_arr = np.load(f'{data_path}/test_err_arr.npy')

#%% evaluate
## train
# X
WINDOW = 3
train_err_list = []
for i in range(31-WINDOW):
    sum_ = np.sum(train_err_arr[:,i:i+WINDOW,:], axis=1)
    train_err_list.append(sum_)
train_err_r = np.concatenate([np.min(train_err_list, axis=0), np.max(train_err_list, axis=0), np.mean(train_err_list, axis=0)], axis=1)
# y
train_problem_r = np.max(train_problem_arr, axis=1)

# model train
train_y = (train_problem_r > 0).astype(int)

# grid search
# 파라미터 설정

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'seed': 1015,
    'verbose': 0,
}

params = load_obj('0118')[0]['params']

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
    sum_ = np.sum(test_err_arr[:,i:i+WINDOW,:], axis=1)
    test_err_list.append(sum_)
test_err_r = np.concatenate([np.min(test_err_list, axis=0), np.max(test_err_list, axis=0), np.mean(test_err_list, axis=0)], axis=1)

# predict
test_prob = []
for model in models:
    test_prob.append(model.predict(test_err_r))
test_prob = np.mean(test_prob, axis=0)

submission['problem'] = test_prob.reshape(-1)
submission.to_csv("submit/submit_2.csv", index = False)