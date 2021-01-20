#%% import
import matplotlib
from util import *

font = {'size': 16, 'family':"NanumGothic"}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

data_path = 'data'

NUM_ERR_TYPES = 42

#%% load dataset
train_err_arr = np.load(f'{data_path}/train_err_arr.npy')
train_problem_arr = np.load(f'{data_path}/train_problem_arr.npy')
test_err_arr = np.load(f'{data_path}/test_err_arr.npy')

#%% preprocessing
import multiprocessing
from _functools import partial

# y
train_problem_r = np.max(train_problem_arr, axis=1)

# feature extraction
df_list = []
for i, train_err in tqdm(enumerate(train_err_arr)):
    df = []
    for e in range(NUM_ERR_TYPES):
        df.append(np.array([train_err[:,e], np.array([e] * train_err.shape[0])]).T)
    df = pd.DataFrame(np.concatenate(df, axis=0), columns = ['value', 'id'])
    df['id'] = df['id'].astype(int)
    df_list.append(df)

## test도 똑같이
df_list_test = []
for i, test_err in tqdm(enumerate(test_err_arr)):
    df = []
    for e in range(NUM_ERR_TYPES):
        df.append(np.array([test_err[:,e], np.array([e] * test_err.shape[0])]).T)
    df = pd.DataFrame(np.concatenate(df, axis=0), columns = ['value', 'id'])
    df['id'] = df['id'].astype(int)
    df_list_test.append(df)

import tsfresh
from tsfresh.feature_extraction.settings import MinimalFCParameters, EfficientFCParameters

tf_list = []
for df in tqdm(df_list):
    if __name__ == '__main__':
        partial_func = partial(tsfresh.extract_features, column_id='id', n_jobs=1, default_fc_parameters=EfficientFCParameters())
        pool = multiprocessing.Pool(processes=6)
        tf = pool.map(partial_func, df)
        tf_list.append(tf)

tf_list_test = []
for df in tqdm(df_list_test):
    if __name__ == '__main__':
        partial_func = partial(tsfresh.extract_features, column_id='id', n_jobs=1, default_fc_parameters=EfficientFCParameters())
        pool = multiprocessing.Pool(processes=6)
        tf = pool.map(partial_func, df)
        tf_list_test.append(tf)

#%% model train and evaluate
train_y = (train_problem_r > 0).astype(int)
# 파라미터 설정
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'seed': 1015,
    'verbose': 1,
}
models, valid_probs = train_model(train_err_r, train_y, params)

# evaluate
threshold = 0.5
valid_preds = np.where(valid_probs > threshold, 1, 0)

# cal scores
evaluate(valid_probs, train_y)
