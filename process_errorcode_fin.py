#%% import
import pandas as pd
import numpy as np
from functools import partial
import pickle
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.metrics import *
from sklearn.model_selection import KFold
import lightgbm as lgb
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction.settings import EfficientFCParameters
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, STATUS_FAIL
import os

data_path = 'data/'
data_save_path = 'data_use/'
param_save_path = 'params/'

if not os.path.isdir(data_save_path):
    os.mkdir(data_save_path)

if not os.path.isdir(param_save_path):
    os.mkdir(param_save_path)

TRAIN_ID_LIST = list(range(10000, 25000))
TEST_ID_LIST = list(range(30000, 45000-1))
NUM_CPU = multiprocessing.cpu_count()
N_USER_TRAIN = len(TRAIN_ID_LIST)
N_USER_TEST = len(TEST_ID_LIST)
N_ERRTYPE = 42


#%% 필요 함수 정의
def process_errcode(user_id, err_df):
    '''
    user_id와 error_data를 입력으로 받아
    하루 단위 데이터로 transform하는 함수

    user_id개의 dictionary를 return하며
    해당 dictionary는 key가 day (1~30) 이고 value가 해당 user, day에 해당하는 dataframe
    '''
    # print(user_id)
    idx = err_df['user_id'] == user_id
    err_df_sub = err_df.loc[idx, :]
    err_in_day = dict()

    if idx.sum() == 0:
        return err_in_day

    for day in range(1, 31):
        idx_2 = err_df_sub['time'].dt.day == day
        err_in_day[day] = err_df_sub.loc[idx_2, 'model_nm' : 'errcode'].reset_index(drop=True)
    return err_in_day


def process_errordata_in_day():
    '''
    load raw error_data and transfrom it to day by day dataframe
    '''
    # train
    train_err_df = pd.read_csv(data_path + 'train_err_data.csv')
    train_err_df.drop_duplicates(inplace=True)
    train_err_df = train_err_df.reset_index(drop=True)
    train_err_df['time'] = pd.to_datetime(train_err_df['time'], format='%Y%m%d%H%M%S')

    err_type_code_train = []
    for user_id in TRAIN_ID_LIST:
        err_type_code_train.append(process_errcode(user_id, err_df= train_err_df))
    del train_err_df

    # save FIXME: 제출시 save가 아니고 바로 사용하도록 바꿔야 함
    with open(f'{data_save_path}err_type_code_train.pkl', 'wb') as f:
        pickle.dump(err_type_code_train, f)
    del err_type_code_train

    # test
    test_err_df = pd.read_csv(data_path + 'test_err_data.csv')
    test_err_df.drop_duplicates(inplace=True)
    test_err_df = test_err_df.reset_index(drop=True)
    test_err_df['time'] = pd.to_datetime(test_err_df['time'], format='%Y%m%d%H%M%S')

    err_type_code_test = []
    for user_id in TEST_ID_LIST:
        err_type_code_test.append(process_errcode(user_id, err_df=test_err_df))
    del test_err_df

    # save
    with open(f'{data_save_path}err_type_code_test.pkl', 'wb') as f:
        pickle.dump(err_type_code_test, f)
    del err_type_code_test


def processing_errcode(errortype, errorcode):
    '''
    errortype과 errorcode를 조합해서 새로운 error type을 생성
    '''
    new_errtype = errortype.copy().astype(str)
    for e in range(1, 43):
        idx = errortype == e

        # 값이 없을 경우
        if idx.sum() == 0:
            continue

        # 값이 있을 경우
        errcode = errorcode[idx]

        # 공백처리 및 다른 처리..
        for i, ec in enumerate(errcode.copy()):
            ec = ec.strip()  # 앞뒤로 공백 제거
            ec = ec.replace('_', '-')
            if '.' in ec:
                ec= ec.split('.')[0]
            if ec.isdigit():
                errcode[i] = str(int(ec))
            errcode[i] = ec

        if e == 1:
            for i, ec in enumerate(errcode):
                if ec == '0':
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + ec
                elif ec[0:2] == 'P-':
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + ec
                else:
                    print(f'Unknown error code for error type {e}:: {ec}')
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'UNKNOWN'
        elif e in [2, 4, 31, 37, 39, 40]:
            # 0과 1만 valid
            idx_unknown = (errcode != '0') * (errcode != '1')
            if idx_unknown.sum() != 0:
                print(f'Unknown error code for error type {e}:: {errcode[idx_unknown]}')
            new_errtype[np.where(idx)[0][idx_unknown]] = str(e) + '-' + 'UNKNOWN'
            new_errtype[np.where(idx)[0][~idx_unknown]] = [str(e) + '-' + s for s in errcode[~idx_unknown]]
        elif e == 3:
            # 0, 1, 2만 valid
            idx_unknown = (errcode != '0') * (errcode != '1') * (errcode != '2')
            if idx_unknown.sum() != 0:
                print(f'Unknown error code for error type {e}:: {errcode[idx_unknown]}')
            new_errtype[np.where(idx)[0][idx_unknown]] = str(e) + '-' + 'UNKNOWN'
            new_errtype[np.where(idx)[0][~idx_unknown]] = [str(e) + '-' + s for s in errcode[~idx_unknown]]
        elif e == 30:
            # 0, 1, 2, 3, 4만 valid
            idx_unknown = (errcode != '0') * (errcode != '1') * (errcode != '2') * (errcode != '3') * (errcode != '4')
            if idx_unknown.sum() != 0:
                print(f'Unknown error code for error type {e}:: {errcode[idx_unknown]}')
            new_errtype[np.where(idx)[0][idx_unknown]] = str(e) + '-' + 'UNKNOWN'
            new_errtype[np.where(idx)[0][~idx_unknown]] = [str(e) + '-' + s for s in errcode[~idx_unknown]]
        elif e == 5:
            for i, ec in enumerate(errcode):
                if ec[0:2] in ['Y-', 'V-', 'U-', 'S-', 'Q-', 'P-', 'M-', 'J-', 'H-', 'E-', 'D-', 'C-', 'B-','En']:
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + ec
                elif ec == 'http':
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'http'
                elif ec.isdigit():
                    if int(ec) in [0, 1]:
                        new_errtype[np.where(idx)[0][i]] = str(e) + '-' + ec
                    else:
                        new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'UNKNOWN'
                else:
                    print(f'UNKNOWN error code for type {e} :: {ec}')
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'UNKNOWN'
        elif e == 8:
            for i, ec in enumerate(errcode):
                if ec =='20':
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + '20'
                elif ec in ['PHONE-ERR', 'PUBLIC-ERR']:
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + ec
                else:
                    print(f'UNKNOWN error code for type {e} :: {ec}')
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'UNKNOWN'
        elif e == 9:
            for i, ec in enumerate(errcode):
                if ec == '1':
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-1'
                elif ec[0:2] in ['C-', 'V-']:
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + ec
                else:
                    print(f'UNKNOWN error code for type {e} :: {ec}')
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-UNKNOWN'
        elif e in [10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 24, 26, 27, 28, 35]:
            # 1만 valid
            idx_unknown = (errcode != '1')
            if idx_unknown.sum() != 0:
                print(f'Unknown error code for error type {e}:: {errcode[idx_unknown]}')
            new_errtype[np.where(idx)[0][idx_unknown]] = str(e) + '-' + 'UNKNOWN'
            new_errtype[np.where(idx)[0][~idx_unknown]] = [str(e) + '-' + s for s in errcode[~idx_unknown]]
        elif e == 14:
            # 1만 valid
            idx_unknown = (errcode != '1') * (errcode != '13') * (errcode != '14')
            if idx_unknown.sum() != 0:
                print(f'Unknown error code for error type {e}:: {errcode[idx_unknown]}')
            new_errtype[np.where(idx)[0][idx_unknown]] = str(e) + '-' + 'UNKNOWN'
            new_errtype[np.where(idx)[0][~idx_unknown]] = [str(e) + '-' + s for s in errcode[~idx_unknown]]
        elif e == 17:
            # 1만 valid
            idx_unknown = (errcode != '1') * (errcode != '12') * (errcode != '13') * (errcode != '14') * (errcode != '21')
            if idx_unknown.sum() != 0:
                print(f'Unknown error code for error type {e}:: {errcode[idx_unknown]}')
            new_errtype[np.where(idx)[0][idx_unknown]] = str(e) + '-' + 'UNKNOWN'
            new_errtype[np.where(idx)[0][~idx_unknown]] = [str(e) + '-' + s for s in errcode[~idx_unknown]]
        elif e == 25:
            for i, ec in enumerate(errcode):
                if 'UNKNOWN' in ec:
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'UNKNOWN'
                elif 'fail' in ec:
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'fail'
                elif 'timeout' in ec:
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'timeout'
                elif 'cancel' in ec:
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'cancel'
                elif 'terminate' in ec:
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'terminate'
                elif ec.isdigit():
                    if int(ec) in [1, 2]:
                        new_errtype[np.where(idx)[0][i]] = str(e) + '-' + ec
                    else:
                        new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'UNKNOWN'
                else:
                    print(f'Unknown error code for error type {e}:: {ec}')
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'UNKNOWN'
        elif e == 32:
            new_errtype[idx] = 'UNKNOWN'
        elif e == 33:
            idx_unknown = (errcode != '1') * (errcode != '2') * (errcode != '3')
            if idx_unknown.sum() != 0:
                print(f'Unknown error code for error type {e}:: {errcode[idx_unknown]}')
            new_errtype[np.where(idx)[0][idx_unknown]] = str(e) + '-' + 'UNKNOWN'
            new_errtype[np.where(idx)[0][~idx_unknown]] = [str(e) + '-' + s for s in errcode[~idx_unknown]]
        elif e == 34:
            idx_unknown = (errcode != '1') * (errcode != '2') * (errcode != '3') * (errcode != '4') * (errcode != '5') * (errcode != '6')
            if idx_unknown.sum() != 0:
                print(f'Unknown error code for error type {e}:: {errcode[idx_unknown]}')
            new_errtype[np.where(idx)[0][idx_unknown]] = str(e) + '-' + 'UNKNOWN'
            new_errtype[np.where(idx)[0][~idx_unknown]] = [str(e) + '-' + s for s in errcode[~idx_unknown]]
        elif e == 36:
            idx_unknown = (errcode != '8')
            if idx_unknown.sum() != 0:
                print(f'Unknown error code for error type {e}:: {errcode[idx_unknown]}')
            new_errtype[np.where(idx)[0][idx_unknown]] = str(e) + '-' + 'UNKNOWN'
            new_errtype[np.where(idx)[0][~idx_unknown]] = [str(e) + '-' + s for s in errcode[~idx_unknown]]
        elif e == 38:
            new_errtype[idx] = 'UNKNOWN'
        elif e == 41:
            idx_unknown = (errcode != 'NFANDROID2')
            if idx_unknown.sum() != 0:
                print(f'Unknown error code for error type {e}:: {errcode[idx_unknown]}')
            new_errtype[np.where(idx)[0][idx_unknown]] = str(e) + '-' + 'UNKNOWN'
            new_errtype[np.where(idx)[0][~idx_unknown]] = [str(e) + '-' + s for s in errcode[~idx_unknown]]
        elif e == 42:
            idx_unknown = (errcode != '2') * (errcode != '3')
            if idx_unknown.sum() != 0:
                print(f'Unknown error code for error type {e}:: {errcode[idx_unknown]}')
            new_errtype[np.where(idx)[0][idx_unknown]] = str(e) + '-' + 'UNKNOWN'
            new_errtype[np.where(idx)[0][~idx_unknown]] = [str(e) + '-' + s for s in errcode[~idx_unknown]]
        else:
            new_errtype[idx] = 'UNKNOWN'

    return new_errtype # 정상적인 return


def generate_new_errtype():
    '''
    error code를 encoding 하기 위한 데이터 생성
    '''
    err_df = pd.read_csv(data_path + 'train_err_data.csv')

    err_df = err_df.loc[:, 'errtype':'errcode']
    # make unique pairs
    err_df.drop_duplicates(inplace=True)
    err_df = err_df.reset_index(drop=True)

    errortype = err_df['errtype'].values
    errorcode = err_df['errcode'].values.astype(str)

    new_errtype = processing_errcode(errortype, errorcode)
    new_errtype = np.unique(new_errtype)
    return new_errtype


def transform_errtype(data):
    '''
    일별 error data의 error type과 code를 조합하여 새로운 errorcode로 변경
    생성된 error code와 기존의 error type을 concatenate
    '''
    err_code = np.zeros((30, N_NEW_ERRTYPE))
    err_type = np.zeros((30, N_ERRTYPE))
    errtype_38_errcode_sum = np.zeros((30,1), dtype=int)
    for day in range(1, 31):
        # error가 없는 user를 skip한다
        if data == {}:
            print('Unknown user, skip it')
            break
        # error code 관련
        transformed_errcode = processing_errcode(data[day]['errtype'].values,
                                                 data[day]['errcode'].values.astype(str))

        # 38만 따로 처리
        idx_38 = data[day]['errtype'].values == 38
        errtype_38_errcode_sum[day-1] = np.sum(data[day]['errcode'].values[idx_38].astype(int))

        try:
            transformed_errcode = encoder.transform(transformed_errcode)
        except ValueError or KeyError:  # 새로운 error code가 있는 경우 valid한 값만 남김
            valid_errcode = []
            for i, errcode in enumerate(transformed_errcode):
                if errcode in new_errtype:
                    valid_errcode.append(errcode)
                else:
                    if 'UNKNOWN' not in errcode:
                        print(f'Skip error code {errcode}')
            # replace
            transformed_errcode = encoder.transform(valid_errcode)
        v, c = np.unique(transformed_errcode, return_counts=True)
        if v.size == 0:
            continue
        err_code[day - 1][v] += c

        # error type 관련
        errtype = data[day][
                      'errtype'].values - 1  # error type이 1~42이므로 index로 바꾸기 위해 1을 빼줌
        v, c = np.unique(errtype, return_counts=True)
        if v.size == 0:
            continue
        err_type[day - 1][v] += c
    err = np.concatenate([err_code, err_type, errtype_38_errcode_sum], axis=1)
    return err


def transform_model_nm(data):
    model_nm = np.zeros((30, 9), dtype = int)
    for day in range(1, 31):
        if data == {}:
            print('Unknown user, skip it')
            break
        for model in np.unique(data[day]['model_nm']):
            model_nm[day-1, int(model[-1])] = 1
    return model_nm


def transform_fwver(data):
    fwver = np.zeros((30, 3), dtype=int)  # 00. 00. 00로 분류
    fwver = fwver.astype(str)

    for day in range(1, 31):
        if data == {}:
            print('Unknown user, skip it')
            break
        for fwver_u in np.unique(data[day]['fwver']):
            striped_fwver = fwver_u.split('.')
            for i in range(len(striped_fwver)):
                fwver[day - 1, i] = striped_fwver[i]
    return fwver


def transform_error_data():
    '''
    dataframe에서 array로 변경
    1. error type and code
    2. model_nm
    3. fwver
    '''
    #### train
    with open(f'{data_save_path}err_type_code_train.pkl', 'rb') as f:
        err_type_code_train = pickle.load(f)

    # error type과 code를 조합한 것으로 transform
    data_list = [err_type_code_train[user_idx] for user_idx in range(N_USER_TRAIN)]
    train_err_list, model_list, fwver_list = [], [], []
    for data in tqdm(data_list):
        train_err_list.append(transform_errtype(data))
    #     model_list.append(transform_model_nm(data))
    #     fwver_list.append(transform_fwver(data))


    # list to array
    train_err_code = np.array(train_err_list)
    # train_models = np.array(model_list)
    # train_fwvers = np.array(fwver_list)

    # save
    np.save(f'{data_save_path}train_err_code_w_38.npy', train_err_code)
    # np.save(f'{data_save_path}train_err_type.npy', train_err_code)
    # np.save(f'{data_save_path}train_models.npy', train_models)
    # np.save(f'{data_save_path}train_fwvers.npy', train_fwvers)

    #### test
    with open(f'{data_save_path}err_type_code_test.pkl', 'rb') as f:
        err_type_code_test = pickle.load(f)

    # error code 관련
    # error type과 code를 조합한 것으로 transform
    data_list = [err_type_code_test[user_idx] for user_idx in range(N_USER_TEST)]
    test_err_list, model_list, fwver_list = [], [], []
    for data in tqdm(data_list):
        test_err_list.append(transform_errtype(data))
    #     model_list.append(transform_model_nm(data))
    #     fwver_list.append(transform_fwver(data))

    # list to array
    test_err_code = np.array(test_err_list)
    # test_models = np.array(model_list)
    # test_fwvers = np.array(fwver_list)

    # save
    np.save(f'{data_save_path}test_err_code_w_38.npy', test_err_code)
    # np.save(f'{data_save_path}test_models.npy', test_models)
    # np.save(f'{data_save_path}test_fwvers.npy', test_fwvers)

    # FIXME: save 말고 바로 return으로 바꿔야 함

from tsfresh.feature_extraction.feature_calculators \
    import fft_aggregated, fft_coefficient, agg_linear_trend, number_crossing_m, skewness, fourier_entropy, cwt_coefficients, cid_ce, \
        symmetry_looking

def tsfresh_manually(train_err_r):
    def func1(func, data, list_):
        f1 = list(func(data, list_))
        f1 = pd.DataFrame(f1)
        f1 = pd.DataFrame(columns=f1[0], data=f1[1].values.reshape(1, -1))
        return f1
    data = []
    for i in tqdm(range(train_err_r.shape[0])):
        tmp = []
        for j in range(N_ERRTYPE + N_NEW_ERRTYPE):
            f1 = func1(fft_aggregated, train_err_r[i, :, j],
                       [{'aggtype': 'centroid'}, {'aggtype': 'variance'}, {'aggtype': 'skew'}, {'aggtype': 'kurtosis'}])
            f2 = func1(fft_coefficient, train_err_r[i, :, j],
                       [{'coeff': 1, 'attr': 'real'}, {'coeff': 1, 'attr': 'imag'}, {'coeff': 1, 'attr': 'abs'},
                        {'coeff': 1, 'attr': 'angle'}])
            f3 = func1(agg_linear_trend, train_err_r[i, :, j], [{'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'var'},
                                                                {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'mean'},
                                                                {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'max'}])
            f4 = func1(symmetry_looking, train_err_r[i, :, j], [{'r': 0.35}])

            # t1 = number_crossing_m(train_err_r[i, :, j], 0)
            # t2 = skewness(train_err_r[i, :, j])
            # t3 = fourier_entropy(train_err_r[i, :, j], 1)
            # t4 = cid_ce(train_err_r[i, :, j], True)
            # t_nu = pd.DataFrame(data=np.array([t1, t2, t3, t4]).reshape(1, -1),
            #                     columns=['number_crossing_m', 'skewness', 'fourier_entropy', 'cid_ce'])

            # feature_for_one_user = pd.concat([f1, f2, f3, f4, t_nu], axis=1)
            feature_for_one_user = pd.concat([f1, f2, f3, f4], axis=1)
            feature_for_one_user.columns = [str(j) + '_' + c for c in feature_for_one_user.columns]
            tmp.append(feature_for_one_user)

        tmp = pd.concat(tmp, axis=1)
        data.append(tmp)

    features = pd.concat(data, axis=0).reset_index(drop=True)
    return features


def extract_err_manually(data, WINDOW = 3): # 임시로는 mean 등이고library 활용할 것
    err_list = []
    for i in range(31 - WINDOW):
        sum_ = np.sum(data[:, i:i + WINDOW, :], axis=1)
        err_list.append(sum_)
    # tsfresh
    # err_r_raw = np.array(err_list).transpose(1,0,2)
    # tsfresh_features = tsfresh_manually(err_r_raw)
    # manually
    err_r = np.concatenate(
        [np.min(err_list, axis=0), np.max(err_list, axis=0), np.mean(err_list, axis=0),
         np.median(err_list, axis=0)], axis=1)
    err_df = pd.DataFrame(err_r)
    return err_df


def extract_err_library(data, WINDOW = 3):
    err_list = []
    for i in range(31 - WINDOW):
        sum_ = np.sum(data[:, i:i + WINDOW, :], axis=1)
        err_list.append(sum_)
    err_r = np.array(err_list).transpose(1,0,2)

    df_list = []
    for i, err in tqdm(enumerate(err_r)):
        df_list.append(np.append(np.array([i] * err.shape[0]).reshape(-1, 1), err, axis=1))
    df_list = pd.DataFrame(np.concatenate(df_list, axis=0), columns=['id'] + list(range(N_ERRTYPE + N_NEW_ERRTYPE)))
    df_list['id'] = df_list['id'].astype(int)

    tf_data = []
    for i in range(N_ERRTYPE + N_NEW_ERRTYPE):
        df_list_sub = df_list.loc[:,['id',i]]
        tf_data_sub = extract_features(df_list_sub, column_id='id',
                                default_fc_parameters=EfficientFCParameters())
        tf_data.append(tf_data_sub)
    tf_data = pd.concat(tf_data, axis=1)
    return tf_data


def extract_model_nm(data, id_list):
    model_sorted_order = np.array([4, 1, 0, 2, 8, 5, 3, 7, 6]) + 1
    # model_diff flag
    model_diff = np.zeros((len(id_list)), dtype=int)

    for i in range(len(id_list)):
        model_diff[i] = (data[i].sum(axis=0) > 0).sum()
    # model_start flag
    model_start = np.zeros((len(id_list)), dtype=int)
    for i in range(len(id_list)):
        occur_idx = np.where(data[i].sum(axis=1))[0]
        if occur_idx.size == 0:
            continue
        first_occur_idx = occur_idx[0]
        model_class = np.where(data[i][first_occur_idx,:])[0] + 1 # 없는것과 0을 구분하기 위해 1을 더해준다
        if type(model_class) == np.ndarray:
            model_start[i] = model_class[0]
        else:
            model_start[i] = model_class
    # model_end flag
    model_end = np.zeros((len(id_list)), dtype=int)
    for i in range(len(id_list)):
        occur_idx = np.where(data[i].sum(axis=1))[0]
        if occur_idx.size == 0:
            continue
        last_occur_idx = occur_idx[-1]
        model_class = np.where(data[i][last_occur_idx, :])[0] + 1  # 없는것과 0을 구분하기 위해 1을 더해준다
        if type(model_class) == np.ndarray:
            model_end[i] = model_class[0]
        else:
            model_end[i] = model_class
    # model 유무
    model_exist = np.zeros((len(id_list), 9), dtype=int)
    for i in range(len(id_list)):
        model_exist[i, :] = data[i].sum(axis=0) > 0

    # model upgrade
    model_upgrade = np.zeros((len(id_list)), dtype=int)
    model_downgrade = np.zeros((len(id_list)), dtype=int)
    for i in range(len(id_list)):
        if model_start[i] == 0 or model_end[i] == 0: # unknown user
            continue
        idx1 = np.where(model_start[i] == model_sorted_order)[0][0]
        idx2 = np.where(model_end[i] == model_sorted_order)[0][0]
        if idx1 == idx2:
            pass
        elif idx1 < idx2:
            model_upgrade[i] = 1
        else:
            model_downgrade[i] = -1

    model_df = pd.DataFrame(data = model_exist)
    model_df['model_diff'] = model_diff
    model_df['model_start'] = pd.Series(model_start, dtype='category')
    model_df['model_end'] = pd.Series(model_end, dtype='category')
    model_df['model_upgrade'] = model_upgrade
    model_df['model_downgrade'] = model_downgrade
    return model_df


def extract_fwver(data, id_list):
    data = data.astype(int)
     # model diff
    fwver_diff = np.zeros((len(id_list), 3), dtype=int)
    fwver_start = np.zeros((len(id_list), 3), dtype=int)
    fwver_end = np.zeros((len(id_list), 3), dtype=int)
    fwver_upgrade = np.zeros((len(id_list), 3), dtype=int)
    fwver_downgrade = np.zeros((len(id_list), 3), dtype=int)

    for i in range(len(id_list)):
        for j in range(3):
            data_sub = data[i][:,j]
            data_sub = data_sub[data_sub != 0]
            # diff
            fwver_diff[i, j] = np.max([len(np.unique(data_sub)) - 1, 0])
            occur_idx = np.where(data_sub)[0]
            if occur_idx.size == 0:
                continue
            first_occur_idx = occur_idx[0]
            fwver_class_start = data_sub[first_occur_idx]
            fwver_start[i, j] = fwver_class_start

            last_occur_idx = occur_idx[-1]
            fwver_class_end = data_sub[last_occur_idx]
            fwver_end[i, j] = fwver_class_end

            fwver_upgrade[i,j] = fwver_class_start < fwver_class_end
            fwver_downgrade[i,j] = (fwver_class_start > fwver_class_end) * -1

    fwver_df = pd.DataFrame()
    for j in range(3):
        fwver_df['fwver_diff_'+str(j)] = fwver_diff[:,j]
        fwver_df['fwver_start_' + str(j)] = pd.Series(fwver_start[:, j])
        fwver_df['fwver_end_' + str(j)] = pd.Series(fwver_end[:, j])
        fwver_df['fwver_upgrade_' + str(j)] = pd.Series(fwver_upgrade[:, j])
        fwver_df['fwver_downgrade_' + str(j)] = pd.Series(fwver_downgrade[:, j])
        break
    return fwver_df


def feature_extraction(option = 1):
    '''
    1. Extract error type
    2. Extract model_nm
    *option
    1: manually, 2: extract from library, 3. load saved data extracted from library
    '''

    ### 0. load dataset
    if option != 3:
        # train_err_arr = np.load(f'{data_save_path}train_err_code_v2.npy')
        # test_err_arr = np.load(f'{data_save_path}test_err_code_v2.npy')
        train_err_arr = np.load(f'{data_save_path}train_err_code_w_38.npy')
        test_err_arr = np.load(f'{data_save_path}test_err_code_w_38.npy')
        # train_err_arr = np.load(f'{data_save_path}train_err_type.npy')
        # test_err_arr = np.load(f'{data_save_path}test_err_type.npy')

        # train_err_arr = train_err_arr[N_NEW_ERRTYPE:,:]
        # test_err_arr = test_err_arr[N_NEW_ERRTYPE:, :]

    ### 1. extract features based on option
    if option == 1:
        train_err_df = extract_err_manually(train_err_arr, WINDOW = 3)
        test_err_df = extract_err_manually(test_err_arr, WINDOW = 3)
    elif option == 2:
        train_err_df = extract_err_library(train_err_arr)
        test_err_df = extract_err_library(test_err_arr)
        nan_idx = np.any(pd.isnull(pd.concat([train_err_df, test_err_df], axis=0)), axis=0)
        tmp = pd.concat([train_err_df, test_err_df], axis=0)
        train_err_df = train_err_df.loc[:, ~nan_idx]
        test_err_df = tmp.iloc[N_USER_TRAIN:, :].reset_index(drop=True)
        test_err_df = test_err_df.loc[:, ~nan_idx]
        # feature selection
        train_prob = pd.read_csv(data_path + 'train_problem_data.csv')
        train_y = np.zeros((N_USER_TRAIN))
        train_y[train_prob.user_id.unique() - 10000] = 1
        train_err_df = select_features(train_err_df, train_y)
        test_err_df = test_err_df.loc[:,train_err_df.columns]
        train_err_df.to_csv(f'{data_save_path}train_err_df_library.csv')
        test_err_df.to_csv(f'{data_save_path}test_err_df_library.csv')
    if option == 3:
        train_err_df = pd.read_csv(f'{data_save_path}train_err_df_library.csv')
        test_err_df = pd.read_csv(f'{data_save_path}test_err_df_library.csv')

    ### 2. extract features from model_nm
    train_models = np.load(f'{data_save_path}train_models.npy')
    train_model_df = extract_model_nm(train_models, TRAIN_ID_LIST)
    test_models = np.load(f'{data_save_path}test_models.npy')
    test_model_df = extract_model_nm(test_models, TEST_ID_LIST)

    ### 3. extract features from model_nm
    train_fwvers = np.load(f'{data_save_path}train_fwvers.npy')
    train_fwver_df = extract_fwver(train_fwvers, TRAIN_ID_LIST)
    test_fwvers = np.load(f'{data_save_path}test_fwvers.npy')
    test_fwver_df = extract_fwver(test_fwvers, TEST_ID_LIST)

    ### 4. concatenate features
    train_data = pd.concat([train_err_df, train_model_df, train_fwver_df], axis=1)
    test_data = pd.concat([test_err_df, test_model_df, test_fwver_df], axis=1)

    # train_data = pd.concat([train_err_df, train_model_df], axis=1)
    # test_data = pd.concat([test_err_df, test_model_df], axis=1)
    # train_data = train_err_df
    # test_data = test_err_df

    return train_data, test_data


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
        if type(train_x) == pd.DataFrame:
            X = train_x.iloc[train_idx,:]
            valid_x = train_x.iloc[val_idx, :]
        elif type(train_x) == np.ndarray:
            X = train_x[train_idx, :]
            valid_x = train_x[val_idx, :]
        else:
            print('Unknown data type for X')
            return -1, -1
        y = train_y[train_idx]
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
            verbose_eval=False,
        )

        # cal valid prediction
        valid_prob = model.predict(valid_x)
        valid_probs[val_idx] = valid_prob

        models.append(model)

        print(model.best_score['valid_0']['auc'])

    return models, valid_probs


def make_param_int(param, key_names):
    for key, value in param.items():
        if key in key_names:
            param[key] = int(param[key])
    return param


class Bayes_tune_model(object):

    def __init__(self):
        self.random_state = 0
        self.space = {}

    # parameter setting
    def lgb_space(self):
        # LightGBM parameters
        self.space = {
            'objective':                'binary',
            'min_child_weight':         hp.quniform('min_child_weight', 1, 10, 1),
            'learning_rate':            hp.uniform('learning_rate',    0.0001, 0.2),
            'max_depth':                -1,
            'num_leaves':               hp.quniform('num_leaves',       5, 200, 1),
            'min_data_in_leaf':		    hp.quniform('min_data_in_leaf',	10, 200, 1),	# overfitting 안되려면 높은 값
            'reg_alpha':                hp.uniform('reg_alpha',0, 1),
            'reg_lambda':               hp.uniform('reg_lambda',0, 1),
            'colsample_bytree':         hp.uniform('colsample_bytree', 0.01, 1.0),
            'colsample_bynode':		    hp.uniform('colsample_bynode',0.01,1.0),
            'bagging_freq':			    hp.quniform('bagging_freq',	0,20,1),
            'tree_learner':			    hp.choice('tree_learner',	['serial','feature','data','voting']),
            'subsample':                hp.uniform('subsample', 0.01, 1.0),
            'boosting':			        hp.choice('boosting', ['gbdt']),
            'max_bin':			        hp.quniform('max_bin',		5,300,1), # overfitting 안되려면 낮은 값
            "min_sum_hessian_in_leaf":  hp.uniform('min_sum_hessian_in_leaf',       1e-5,1e-1),
            'random_state':             self.random_state,
            'n_jobs':                   -1,
            'metrics':                  'auc',
            'verbose':                  -1,
        }

    # optimize
    def process(self, clf_name, train_set, trials, algo, max_evals):
        fn = getattr(self, clf_name+'_cv')
        space = getattr(self, clf_name+'_space')
        space()
        fmin_objective = partial(fn, train_set=train_set)
        try:
            result = fmin(fn=fmin_objective, space=self.space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials

    def lgb_cv(self, params, train_set):
        params = make_param_int(params, ['max_depth','num_leaves','min_data_in_leaf',
                                     'min_child_weight','bagging_freq','max_bin'])

        train_x, train_y = train_set
        _, valid_probs = train_model(train_x, train_y, params)
        best_loss = roc_auc_score(train_y, valid_probs)
        # Dictionary with information for evaluation
        return {'loss': -best_loss, 'params': params, 'status': STATUS_OK}


#%% main문
if __name__ == '__main__':
    '''
    1. 시계열인 error data를 일별로 변경
    '''
    # process_errordata_in_day()
    print('Process 1 Done')

    '''
    2. error type과 code를 조합하여 새로운 error type 생성
    '''
    new_errtype_tmp = generate_new_errtype()
    # Unknown 제거
    new_errtype = []
    for err in new_errtype_tmp:
        if 'UNKNOWN' not in err:
            new_errtype.append(err)

    N_NEW_ERRTYPE = len(new_errtype)
    # 새로운 error type을 encoding
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(new_errtype)
    print('Process 2 Done')
    '''
    3. 1에서 변경된 데이터를 차례로 processing
    '''
    # transform_error_data()
    print('Process 3 Done')

    # %%+
    '''
    4. 일별 데이터에서 피쳐를 추출하여 학습에 적합한 데이터로 변경 
    *option 
    1: manually
    2: extract from library
    3. load saved data extracted from library
    '''
    # from error data
    train_X, test_X = feature_extraction(option = 1)

    cols = train_X.columns
    train_X.columns = range(train_X.shape[1])
    test_X.columns = range(test_X.shape[1])

    print('Process 4 Done')

    '''
    5. train_problem (y) 생성
    '''
    train_prob = pd.read_csv(data_path + 'train_problem_data.csv')
    train_y = np.zeros((N_USER_TRAIN))
    train_y[train_prob.user_id.unique() - 10000] = 1
    print('Process 5 Done')

    '''
    6. fit
    '''
    param_select_option = input('Param: (1) default, (2) bayes opt, (3) previous best param')
    param_select_option = int(param_select_option)
    # param_select_option = 2
    if param_select_option == 1:
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'seed': 1015,
            'verbose': 0,
        }

    elif param_select_option == 2:
        MAX_EVALS = input('Number of iteration for tuning')
        MAX_EVALS = int(MAX_EVALS)
        # MAX_EVALS = 2500

        bayes_trials = Trials()
        obj = Bayes_tune_model()
        tuning_algo = tpe.suggest  # -- bayesian opt
        # tuning_algo = tpe.rand.suggest # -- random search
        obj.process('lgb', [train_X, train_y],
                    bayes_trials, tuning_algo, MAX_EVALS)

        # save the best param
        best_param = sorted(bayes_trials.results, key=lambda k: k['loss'])[0]['params']
        with open(f'tune_results/0129-v2-local.pkl', 'wb') as f:
            pickle.dump(bayes_trials.results, f, pickle.HIGHEST_PROTOCOL)
        print('Process 6 Done')
        params = best_param

    elif param_select_option == 3:
        from util import load_obj
        params = load_obj('0129-local')[0]['params']

    models, valid_probs = train_model(train_X, train_y, params)

    # evaluate
    threshold = 0.5
    valid_preds = np.where(valid_probs > threshold, 1, 0)
    recall = recall_score(train_y, valid_preds)
    precision = precision_score(train_y, valid_preds)
    auc_score = roc_auc_score(train_y, valid_probs)
     # validation score
    print('Process 7 Done')
    print('Validation score is {:.5f}'.format(auc_score))

#%%
    '''
    8. predict
    '''
    submission = pd.read_csv(data_path + '/sample_submission.csv')

    # predict
    test_prob = []
    for model in models:
        test_prob.append(model.predict(test_X))
    test_prob = np.mean(test_prob, axis=0)

    submission['problem'] = test_prob.reshape(-1)
    submission.to_csv("submission.csv", index=False)
    print('Process 8 Done')