#%%
import pandas as pd
import numpy as np
import matplotlib
font = {'size': 16, 'family':"NanumGothic"}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt

data_path = 'data/'

#%% train_err 데이터
train_err = pd.read_csv(data_path + 'train_err_data.csv')

# 1. time 처리
train_err['time'] = pd.to_datetime(train_err['time'], format='%Y%m%d%H%M%S')

# 2. model_nm: 이상 없음
# np.unique(train_err['model_nm'], return_counts = True)

# 3. fwver 확인: 이상 없음
# np.unique(train_err['fwver'], return_counts=True)

# 4. errtype 확인: 이상 없음
# np.unique(train_err['errtype'], return_counts=True)

# 5. errcode: #FIXME: 분석이 더 필요함

del train_err
#%% train_quality 데이터
import re
from tqdm import tqdm
def string2num(x):
    # (,)( )과 같은 불필요한 데이터 정제
    x = re.sub(r"[^0-9]+", '', str(x))
    if x =='':
        return 0
    else:
        return int(x)
train_quality = pd.read_csv(data_path + 'train_quality_data.csv')

# quality에서 str이 섞여있는 부분은 , 등이 포함되어 있으므로 처리해줌
# nan도 포함되어 있는데 나중에 처리해야함
for i in range(13):
    if train_quality['quality_'+str(i)].dtype == object:
        print(f'====={i}====')
        for j in tqdm(range(train_quality.shape[0])):
            val = train_quality.loc[j,'quality_' + str(i)]
            try:
                int(val)
            except:
                print(val)
                train_quality.loc[j,'quality_' + str(i)] = string2num(val)

# save the processed data
train_quality.to_csv('data/train_quality_data_cleansed.csv', index=False)