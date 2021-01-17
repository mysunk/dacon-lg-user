#%% import
import matplotlib
from util import *

font = {'size': 16, 'family':"NanumGothic"}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

data_path = '../data/'

#%% train data load
train_err = pd.read_csv(data_path + 'train_err_data.csv')
train_quality = pd.read_csv(data_path + 'train_quality_data_cleansed.csv')
train_problem = pd.read_csv(data_path + 'train_problem_data.csv')
train_err['time'] = pd.to_datetime(train_err['time'], format='%Y%m%d%H%M%S')
train_problem['time'] = pd.to_datetime(train_problem['time'], format='%Y%m%d%H%M%S')
train_user_id = np.unique(train_err['user_id'])

print('Train data load done')
# #%% 몇개만 사용
# TRAIN_SPLIT = int(len(train_user_id) * 0.1)
# train_user_id = train_user_id[:TRAIN_SPLIT]
#
# TRAIN_SPLIT_IDX = np.where(train_user_id[-1] == train_err['user_id'])[0][0]
# train_err = train_err.iloc[:TRAIN_SPLIT_IDX,:]
#
# print('Split done')

#%% 시계열 데이터 => 일별로 변환
train_err_arr = np.zeros((len(train_user_id), 30, 42))
train_problem_arr = np.zeros((len(train_user_id), 30))
for i, user_id in enumerate(tqdm(train_user_id)):
    err = process_err(train_err, user_id)
    prob = process_prob(train_problem, user_id)
    train_err_arr[i] = err
    train_problem_arr[i] = prob

del train_err, train_problem, train_quality

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
# 파라미터 설정
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'seed': 1015,
    'verbose': 0,
}
models, valid_probs = train_model(train_err_r, train_y, params)

# evaluate
threshold = 0.5
valid_preds = np.where(valid_probs > threshold, 1, 0)

# cal scores
recall = recall_score(train_y, valid_preds)
precision = precision_score(train_y, valid_preds)
auc_score = roc_auc_score(train_y, valid_probs)
print(auc_score)

#%% test data load
test_err = pd.read_csv(data_path + 'test_err_data.csv')
submission = pd.read_csv(data_path + 'sample_submission.csv')
test_id = np.unique(submission['user_id'])
test_err['time'] = pd.to_datetime(test_err['time'], format='%Y%m%d%H%M%S')

# preprocessing
test_err_arr = np.zeros((len(test_id), 30, 42))
for i, user_id in enumerate(tqdm(test_id)):
    err = process_err(test_err, user_id)
    test_err_arr[i] = err
del test_err

test_err_list = []
for i in range(31-WINDOW):
    sum_ = np.sum(test_err_arr[:,i:i+WINDOW,:], axis=1)
    test_err_list.append(sum_)
train_err_r = np.concatenate([np.min(test_err_list, axis=0), np.max(test_err_list, axis=0), np.mean(test_err_list, axis=0)], axis=1)

# predict
test_prob = []
for model in models:
    test_prob.append(model.predict(train_err_r))
test_prob = np.mean(test_prob, axis=0)

submission['problem'] = test_prob.reshape(-1)
submission.to_csv("../submit/submit_1.csv", index = False)

#%%
# err_arr = np.concatenate(err_arr, axis=0)
# problem_arr = np.concatenate(problem_arr, axis=0)
#
# corrs = []
# for i in range(42):
#     corr = np.corrcoef(err_arr[:, i], problem_arr)[0,1]
#     corrs.append(corr)
#
# print(np.nanmax(corrs))

#%% 분석
# # 1. 일별의 max값을 이용: 0. 22
# print(np.corrcoef(np.max(err_arr, axis=1)[:,25], np.max(problem_arr, axis=1))[0,1])
#
# # 2. n일의 합의 max값을 이용
# data = []
# for i in range(28):
#     sum_ = np.sum(err_arr[:,i:i+2,:], axis=1)
#     data.append(sum_)
#     print(np.corrcoef(sum_[:, 25], np.max(problem_arr, axis=1))[0, 1])
#
# print(np.corrcoef(np.max(data, axis=0)[:,25], np.max(problem_arr, axis=1))[0,1])

#%% save processed data
np.save('../data/train_err_arr.npy', train_err_arr)
np.save('../data/train_problem_arr.npy', train_problem_arr)
np.save('../data/test_err_arr.npy', test_err_arr)
