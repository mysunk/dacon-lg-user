#%% import
import matplotlib
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm

font = {'size': 16, 'family':"NanumGothic"}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

#%%
# with open(f'..data_p/err_type_code_train.pkl', 'rb') as f:
#     err_type_code_train = pickle.load(f)

#%% load raw data
data_path = 'data/'
err_df = pd.read_csv(data_path + 'train_err_data.csv')
err_df.drop_duplicates(inplace = True)
err_df = err_df.reset_index(drop = True)
err_df = err_df.loc[:,'errtype':'errcode']
err_df['err_mod'] = err_df['errtype'].copy()

errortype = err_df['errtype'].values
errorcore = err_df['errcode'].values.astype(str)
error_mod = err_df['err_mod'].values

del err_df

#%% error type과 code를 조합
# err_df = err_type_code_train[user_id_idx][day] # 0번째 사용자가 day1에 발생한 error

def multiprocessing_helper(e):
    print(e)
    idx = err_df['errtype'] == e

    # 값이 없을 경우
    if idx.sum() == 0:
        return err_df['errtype']

    # 값이 있을 경우
    errcode = err_df.loc[idx, 'errcode']
    if e == 1:
        for i, ec in enumerate(errcode):
            if ec == '0':
                err_df.loc[np.where(idx)[0][i], 'err_mod'] = str(e) + '-' + ec
            elif ec[0] == 'P':
                err_df.loc[np.where(idx)[0][i], 'err_mod'] = str(e) + '-' + 'P'
            else:
                print('Unknown error code for error type 1')
                return -1
    elif e in [2, 4, 31, 37, 39, 40]:
        err_df.loc[idx, 'err_mod'] = str(e) + '-' + errcode
    elif e == 3:
        err_df.loc[idx, 'err_mod'] = str(e) + '-' + errcode
    elif e == 30:
        err_df.loc[idx, 'err_mod'] = str(e) + '-' + errcode
    elif e == 5:
        for i, ec in enumerate(errcode):
            if ec[0] in ['Y', 'V', 'U', 'S', 'Q', 'P', 'M', 'J', 'H', 'E', 'D', 'C', 'B']:
                err_df.loc[np.where(idx)[0][i], 'err_mod'] = str(e) + '-' + ec[0]
            else:
                try:
                    int(ec)
                    err_df.loc[np.where(idx)[0][i], 'err_mod'] = str(e) + '-' + 'num'
                except:
                    print('Unknown error code for error type 5: It should be int')
                    print(ec)
                    return -1
    elif e in [6, 7]:
        err_df.loc[idx, 'err_mod'] = str(e) + '-' + errcode
    elif e == 8:
        err_df.loc[idx, 'err_mod'] = str(e) + '-' + errcode
    elif e == 9:
        for i, ec in enumerate(errcode):
            if ec == '1':
                err_df.loc[np.where(idx)[0][i], 'err_mod'] = str(e) + '-' + ec  # 1인 경우
            elif ec[0] in ['C', 'V']:
                err_df.loc[np.where(idx)[0][i], 'err_mod'] = str(e) + '-' + ec[0]
            else:
                print('Unknown error code for error type 9')
                return -1
    elif e in [10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 24, 26, 27, 28, 35]:
        err_df.loc[idx, 'err_mod'] = str(e) + '-' + errcode
    elif e == 14:
        err_df.loc[idx, 'err_mod'] = str(e) + '-' + errcode
    elif e == 17:
        err_df.loc[idx, 'err_mod'] = str(e) + '-' + errcode
    elif e == 23:
        for i, ec in enumerate(errcode):
            if 'UNKNOWN' in ec:
                err_df.loc[np.where(idx)[0][i], 'err_mod'] = str(e) + '-' + 'UNKNOWN'
            elif 'fail' in ec:
                err_df.loc[np.where(idx)[0][i], 'err_mod'] = str(e) + '-' + 'fail'
            elif 'timeout' in ec:
                err_df.loc[np.where(idx)[0][i], 'err_mod'] = str(e) + '-' + 'timeout'
            elif 'active' in ec:
                err_df.loc[np.where(idx)[0][i], 'err_mod'] = str(e) + '-' + 'active'
            elif 'standby' in ec:
                err_df.loc[np.where(idx)[0][i], 'err_mod'] = str(e) + '-' + 'standby'
            elif 'terminate' in ec:
                err_df.loc[np.where(idx)[0][i], 'err_mod'] = str(e) + '-' + 'terminate'
            else:
                print('Unknown error code for error type 23')
                return -1
    elif e == 25:
        for i, ec in enumerate(errcode):
            if 'UNKNOWN' in ec:
                err_df.loc[np.where(idx)[0][i], 'err_mod'] = str(e) + '-' + 'UNKNOWN'
            elif 'fail' in ec:
                err_df.loc[np.where(idx)[0][i], 'err_mod'] = str(e) + '-' + 'fail'
            elif 'timeout' in ec:
                err_df.loc[np.where(idx)[0][i], 'err_mod'] = str(e) + '-' + 'timeout'
            elif 'cancel' in ec:
                err_df.loc[np.where(idx)[0][i], 'err_mod'] = str(e) + '-' + 'cancel'
            elif 'terminate' in ec:
                err_df.loc[np.where(idx)[0][i], 'err_mod'] = str(e) + '-' + 'terminate'
            else:
                try:
                    int(ec)
                    err_df.loc[np.where(idx)[0][i], 'err_mod'] = str(e) + '-' + 'num'
                except:
                    print('Unknown error code for error type 25: It should be int')
                    print(ec)
                    return -1
    elif e == 32:
        pass
    elif e == 33:
        err_df.loc[idx, 'err_mod'] = str(e) + '-' + errcode
    elif e == 34:
        err_df.loc[idx, 'err_mod'] = str(e) + '-' + errcode
    elif e == 36:
        err_df.loc[idx, 'err_mod'] = str(e) + '-' + errcode
    elif e == 38:
        pass
    elif e == 41:
        err_df.loc[idx, 'err_mod'] = str(e) + '-' + errcode
    elif e == 42:
        err_df.loc[idx, 'err_mod'] = str(e) + '-' + errcode
    else:
        print('Unknown error type')
        return -1

    return err_df['err_mod'] # 정상적인 return

e = list(range(1, 43))
import multiprocessing
from multiprocessing import Pool

if __name__ == '__main__':
    NUM_CPU = multiprocessing.cpu_count()
    with Pool(NUM_CPU) as p:
        err_mod_list = p.map(multiprocessing_helper, e)

    print('Done processing')
    with open(f'err_mod_list.pkl', 'wb') as f:
        pickle.dump(err_mod_list, f)