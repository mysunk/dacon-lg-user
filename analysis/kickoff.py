#%%
import pandas as pd
import numpy as np
import matplotlib
font = {'size': 16, 'family':"NanumGothic"}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt

data_path = 'data/'

#%% 데이터 로드
train_err = pd.read_csv(data_path + 'train_err_data.csv')
train_quality = pd.read_csv(data_path + 'train_quality_data_cleansed.csv')
train_problem = pd.read_csv(data_path + 'train_problem_data.csv')

#%% 1. train_err
i = 6
model = 'model_'+str(i)
idx = train_err['model_nm'] == model
print(np.unique(train_err.loc[idx, 'errtype']))

v, c = np.unique(train_err['errtype'], return_counts = True)

## 에러 타입과 발생 빈도
idx = np.argsort(c)
idx = idx[::-1]
NUM_SHOW = 20
plt.figure(figsize=(10,3))
plt.title('에러 타입과 발생 빈도')
plt.bar(range(NUM_SHOW), c[idx[:NUM_SHOW]])
plt.xticks(range(NUM_SHOW), v[idx[:NUM_SHOW]])
plt.xlabel('Error type')
plt.ylabel('Occurrence')
plt.show()

## model_nm과 error type 관계
idx = train_err['errtype'] == 2
print(np.unique(train_err.loc[idx, 'model_nm']))

## error code와 error type의 관계
# 1. errorcode 전처리
train_err['errcode'] = train_err['errcode'].astype(str)
idx = train_err['errtype'] == 1
errcode_list = np.unique(train_err.loc[:, 'errcode'])
from collections import defaultdict
errcode_dict = defaultdict(list)
for element in errcode_list:
    try:
        element = int(element)
        errcode_dict['int'].append(element)
    except:
        errcode_dict['str'].append(element)

#%% 2. error type별 error code
for i in range(1,43):
    i = 5
    print(f'{i}')
    idx = train_err['errtype'] == i
    print(np.unique(train_err.loc[idx, 'errcode'], return_counts = True))
    break

#%% 3. fwver에 따른 quality 분포
fwvers = np.unique(train_quality['fwver'].astype(str))

for fwver in fwvers:
    idx = train_quality['fwver'] == fwver
    for i in range(13):
        data = train_quality.loc[idx,'quality_'+str(i)]
        data = data[~pd.isnull(data)]
        plt.hist(data, label=i)
    plt.title(f'fwver: {fwver}')
    plt.xlabel('Quality')
    plt.ylabel('Count')
    plt.legend(loc='upper right')
    plt.show()
#%% 4. fwver가 nan일 때 분석
idx = pd.isnull(train_quality['fwver'])
tmp = train_quality.loc[idx, :]

#%% quality나 fwver에 nan이 하나라도 있으면 버림
idx = pd.isnull(train_quality)
idx = ~np.any(idx, axis=1)
train_quality = train_quality.loc[idx,:].reset_index(drop=True)

#%% 한 user가 여러 fwver를 가지는지?
train_quality['fwver'] = train_quality['fwver'].astype(str)

for user_id in np.unique(train_quality['user_id']):
    idx = train_quality['user_id'] == user_id
    v = np.unique(train_quality.loc[idx, 'fwver'])
    v = list(v)
    if 'nan' in v:
        v.remove('nan')
    if len(v) != 1:
        print(len(v))

#%% fwver에 따른 quality의 variation : ver.4
fwvers = np.unique(train_quality['fwver'].astype(str))
plt.figure()
for i in range(4):
    plt.subplot(4,1,i+1)
    idx = train_quality['fwver'] == fwvers[i]
    val = np.var(train_quality.loc[idx, 'quality_0':'quality_12'], axis=0)
    plt.title(f'fwver: {fwvers[i]}')
    plt.bar(range(13), val, label=fwver)
    plt.yticks([])
    if i != 3:
        plt.xticks([])
    if i == 3:
        plt.xticks(range(13), range(13))
        plt.xlabel('Quality index')
plt.show()

#%% fwver에 따른 quality의 variation : ver.5
plt.figure(figsize=(5,10))
for i in range(4, 10):
    plt.subplot(6,1,i-3)
    idx = train_quality['fwver'] == fwvers[i]
    val = np.var(train_quality.loc[idx, 'quality_0':'quality_12'], axis=0)
    plt.title(f'fwver: {fwvers[i]}')
    plt.bar(range(13), val, label=fwver)
    plt.yticks([])
    if i != 9:
        plt.xticks([])
    if i == 9:
        plt.xticks(range(13), range(13))
        plt.xlabel('Quality index')
plt.show()

#%% fwver의 첫번째 version과 quality간의 correlation
nan_idx = (pd.isnull(train_quality)).sum(axis=1) > 0
data = train_quality.loc[~nan_idx,:].reset_index(drop=True)

version = np.array([data['fwver'][i][:2] for i in range(data.shape[0])])

for i in range(13):
    print(np.corrcoef((version == '05').astype(int),
                data['quality_'+str(i)].values))

#%% quality간의 상관관계

for i in range(12):
    for j in range(i+1,12):
        print(f'i, j in {i}, {j}')
        print('{:.2f}'.format(np.corrcoef(data['quality_'+str(i)].values,
                        data['quality_'+str(j)].values)[0,1]))

#%% 하나의 user_id에 대하여
idx = train_err['user_id'] == 10000
err_data = train_err.loc[idx,:].copy()
del train_err

