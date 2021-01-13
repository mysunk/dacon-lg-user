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
train_quality = pd.read_csv(data_path + 'train_quality_data_cleansed.csv')
train_problem = pd.read_csv(data_path + 'train_problem_data.csv')
train_err['time'] = pd.to_datetime(train_err['time'], format='%Y%m%d%H%M%S')

print('Train data load done')

#%% train-val split
train_user_id = np.unique(train_err['user_id'])
TRAIN_SPLIT = int(len(train_user_id) * 0.1)
train_user_id_tr = train_user_id[:TRAIN_SPLIT]
train_user_id_val = train_user_id[TRAIN_SPLIT:]

### 임시로..속도때문에
# FIXME
SPLIT_IDX = np.where(train_user_id_val[0] == train_err['user_id'])[0][0]
train_err = train_err.iloc[:SPLIT_IDX,:]

#%% custom function
def process_errcode(errcode):
    errcode_p = np.zeros(errcode.shape, dtype=int)
    success_idx = (errcode == '0') | (errcode == 0)
    errcode_p[~success_idx] = 1
    return errcode_p

train_err['errcode'] = process_errcode(train_err['errcode'])

#%% 전체 error type에 대하여 기간 최적화
# window length [hour]
w_b = 24*3
w_a = 24*6

train_problem_count = dict()
train_err_count = dict()
for user_id in tqdm(train_user_id_tr):

    # find data corresponding to user_id
    idx_err = train_err['user_id'] == user_id
    # idx_quality = train_quality['user_id'] == user_id
    idx_problem = train_problem['user_id'] == user_id

    # strip data
    train_err_u = train_err.loc[idx_err, :]
    # train_quality_u = train_quality.loc[idx_quality, :]
    train_problem_u = train_problem.loc[idx_problem, 'time']

    # make label
    train_problem_count[user_id] = train_problem_u.shape[0]

    # cut train_err w.r.t w_b and w_a
    time_ranges = []
    for i in range(train_problem_u.shape[0]):
        prob_time = train_problem_u.iloc[i]
        # error 범위 한정
        idx_interest_b = \
            (train_err_u['time'] < pd.to_datetime(prob_time, format='%Y%m%d%H%M%S')).values & \
            (train_err_u['time'] > pd.to_datetime(prob_time, format='%Y%m%d%H%M%S') - pd.Timedelta(w_b, unit='h')).values

        idx_interest_a = \
            (train_err_u['time'] > pd.to_datetime(prob_time, format='%Y%m%d%H%M%S')).values & \
            (train_err_u['time'] < pd.to_datetime(prob_time, format='%Y%m%d%H%M%S') + pd.Timedelta(w_a, unit='h')).values

        time_range = idx_interest_b | idx_interest_a
        time_ranges.append(time_range)
    time_ranges = np.array(time_ranges).T
    if len(time_ranges) == 0:
        err_count = np.zeros((42), dtype=int)
        train_err_count[user_id] = err_count
        continue
    time_ranges = np.any(time_ranges, axis=1)
    train_err_u = train_err_u.iloc[time_ranges,-2:]

    err_count = np.zeros((42), dtype=int)
    for i in range(1, NUM_ERR_TYPES+1):
        # errcode 미사용
        err_count[i-1] = (train_err_u['errtype'] == i).sum()
        # errcode를 사용
        # FIXME
        # err_count[i - 1] = ((train_err_u['errtype'] == i).values * (train_err_u['errcode'] != 0).values).sum()
    train_err_count[user_id] = err_count

# del train_err, train_quality, train_problem
print('Preprocessing done')

### 분석
train_err_count = pd.DataFrame.from_dict(train_err_count).T
train_err_count.columns = range(1,43)
train_err_count['sum'] = train_err_count.sum(axis=1)
train_problem_count = pd.DataFrame(train_problem_count,index = [0]).T

corr_results = []
for i in range(train_err_count.shape[1]):
    corr_results.append(np.corrcoef(train_err_count.iloc[:,i], train_problem_count.iloc[:,0])[0,1])
corr_results = np.array(corr_results)
print(np.nanmax(corr_results))

# plt.figure(figsize=(15,5))
plt.title('before: {}시간, after: {}시간 // max: {:.2f}'.format(w_b, w_a, np.nanmax(corr_results)))
feature_name = train_err_count.columns
del_idx = np.isnan(corr_results)
corr_results = corr_results[~del_idx]
feature_name = feature_name[~del_idx]
idx = np.argsort(corr_results)[::-1][:5]

plt.bar(range(5), corr_results[idx])
plt.xlabel('Error type')
plt.ylabel('Correlation')
plt.xticks(range(5),feature_name[idx], rotation=30)
plt.ylim(0.5, 0.9)
plt.show()

#%% error type에 따라 기간 최적화
train_problem_count = dict()
train_err_count = dict()
w_a = 144
for user_id in tqdm(train_user_id_tr):
    # find data corresponding to user_id
    idx_err = train_err['user_id'] == user_id
    # idx_quality = train_quality['user_id'] == user_id
    idx_problem = train_problem['user_id'] == user_id

    # strip data
    train_err_u = train_err.loc[idx_err, :]
    # train_quality_u = train_quality.loc[idx_quality, :]
    train_problem_u = train_problem.loc[idx_problem, 'time']

    # make label
    train_problem_count[user_id] = train_problem_u.shape[0]

    err_counts = []
    for w_b in range(24, 24 * 5, 24):
        # cut train_err w.r.t w_b and w_a
        time_ranges = []
        for i in range(train_problem_u.shape[0]):
            prob_time = train_problem_u.iloc[i]
            # error 범위 한정
            idx_interest_b = \
                (train_err_u['time'] < pd.to_datetime(prob_time, format='%Y%m%d%H%M%S')).values & \
                (train_err_u['time'] > pd.to_datetime(prob_time, format='%Y%m%d%H%M%S') - pd.Timedelta(w_b, unit='h')).values

            idx_interest_a = \
                (train_err_u['time'] > pd.to_datetime(prob_time, format='%Y%m%d%H%M%S')).values & \
                (train_err_u['time'] < pd.to_datetime(prob_time, format='%Y%m%d%H%M%S') + pd.Timedelta(w_a, unit='h')).values

            time_range = idx_interest_b | idx_interest_a
            time_ranges.append(time_range)
        time_ranges = np.array(time_ranges).T
        if len(time_ranges) == 0:
            err_count = np.zeros((42), dtype=int)
            err_counts.append(err_count)
            continue
        time_ranges = np.any(time_ranges, axis=1)

        # time_range에 해당하는 error수 집계
        err_count = np.zeros((42), dtype=int)
        for i in range(1, NUM_ERR_TYPES+1):
            # errcode 미사용
            err_count[i-1] = (train_err_u['errtype'][time_ranges] == i).sum()
        err_counts.append(err_count)
    train_err_count[user_id] = np.array(err_counts)

train_problem_count = pd.DataFrame(train_problem_count,index = [0]).T

#%% 분석
error_type_idx = 15
data = dict()
for key,value in train_err_count.items():
    data[key] = value[:,error_type_idx]
data = pd.DataFrame(data).T
data.columns = range(24, 24 * 5, 24)

corr_results = []
for i in range(data.shape[1]):
    corr_results.append(np.corrcoef(data.iloc[:,i], train_problem_count.iloc[:,0])[0,1])
corr_results = np.array(corr_results)
print(np.nanmax(corr_results))

# plt.figure(figsize=(15,5))
plt.title('error type: {} // max: {:.2f}'.format(error_type_idx+1, np.nanmax(corr_results)))
feature_name = data.columns
del_idx = np.isnan(corr_results)
corr_results = corr_results[~del_idx]
feature_name = feature_name[~del_idx]
idx = np.argsort(corr_results)[::-1][:data.shape[1]]

plt.bar(range(data.shape[1]), corr_results[idx])
plt.xlabel('w_b')
plt.ylabel('Correlation')
plt.xticks(range(data.shape[1]),feature_name[idx], rotation=30)
plt.ylim(0.5, 0.9)
plt.show()