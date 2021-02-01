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
errcode_save_name = 'err_code_v1_w_38'

transform = False
infer = True

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
N_QUALITY = 13


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
                ec = ec.split('.')[0]
            if ec.isdigit():
                errcode[i] = str(int(ec))
            errcode[i] = ec

        if e == 1:
            for i, ec in enumerate(errcode):
                if ec == '0':
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + ec
                elif ec[0] == 'P':
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'P'
                elif ec.isdigit():
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'num'
                else:
                    print('Unknown error code for error type 9')
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'UNKNOWN'
        elif e == 5:
            for i, ec in enumerate(errcode):
                if ec[0] in ['Y', 'V', 'U', 'S', 'Q', 'P', 'M', 'J', 'H', 'E', 'D', 'C', 'B']:
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + ec[0]
                elif ec == 'nan':
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'UNKNOWN'
                elif ec.isdigit():
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'num'
                else:
                    print(f'UNKNOWN error code for type 5 :: {ec}')
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'UNKNOWN'
        elif e == 8:
            for i, ec in enumerate(errcode):
                if ec.isdigit():
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'num'
                elif ec in ['PHONE-ERR', 'PUBLIC-ERR']:
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + ec
                else:
                    print(f'UNKNOWN error code for type 5 :: {ec}')
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'UNKNOWN'
        elif e == 9:
            for i, ec in enumerate(errcode):
                if ec.isdigit():
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-num'
                elif ec[0] in ['C', 'V']:
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + ec[0]
                else:
                    print('Unknown error code for error type 9')
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-UNKNOWN'
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
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'num'
                else:
                    print('Unknown error code for error type 9')
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'UNKNOWN'
        elif e == 32:
            for i, ec in enumerate(errcode):
                if ec[0] == '-':
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'neg_num'
                else:
                    # if int(ec) >= 78:
                    #     new_errtype[np.where(idx)[0][i]] = str(e) + '-' + ec
                    # else:
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'pos_num'
            # new_errtype[idx] = str(e) + '-num'
        elif e == 38:
            new_errtype[idx] = str(e) + '-num'
        else:
            new_errtype[idx] = np.array([str(e) + '-' + ec for ec in errcode])

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
    errtype_38_errcode_sum = np.zeros((30, 1), dtype=int)
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


def transform_quality_data(data_type):
    '''
    quality data의 dataframe에서 array로 변경
    '''
    if data_type == 'train':
        id_list = TRAIN_ID_LIST
        n_user = N_USER_TRAIN
    elif data_type == 'test':
        id_list = TEST_ID_LIST
        n_user = N_USER_TEST

    quality = pd.read_csv(data_path + f'{data_type}_quality_data.csv')
    quality['time'] = pd.to_datetime(quality['time'], format='%Y%m%d%H%M%S')
    quality.replace(",", "", regex=True, inplace=True)
    quality.loc[:,'quality_0':'quality_12'] = quality.loc[:,'quality_0':'quality_12'].astype(float)

    # drop nan
    drop_idx = np.any(pd.isnull(quality), axis=1)
    quality = quality.loc[~drop_idx,:].reset_index(drop=True)

    # drop -1
    qual_arr = np.zeros((n_user, 31, N_QUALITY))
    count_minus_1 = np.zeros((n_user, 31, N_QUALITY), dtype=int)
    for i, user_id in tqdm(enumerate(id_list)):
        idx = quality['user_id'] == user_id
        if idx.sum() == 0:
            continue
        quality_sub = quality.loc[idx,:]
        for j, day in enumerate([31] + list(range(1, 31))): # 10월 31일 처리
            idx_day = quality_sub['time'].dt.day == day
            # -1 처리
            minus_1_idx = quality_sub.loc[idx_day, 'quality_0':'quality_12'] == -1
            count_minus_1[i, j, :] = minus_1_idx.sum()
            qual_arr[i, j, :] = np.nansum(quality_sub.loc[idx_day,'quality_0':'quality_12'][~minus_1_idx], axis=0)
    qual_arr = np.append(qual_arr, count_minus_1, axis=2)
    # FIXME
    np.save(f'{data_save_path}{data_type}_quality.npy', qual_arr)


def transform_error_data(data_type):
    '''
    dataframe에서 array로 변경
    1. error type and code
    2. model_nm
    3. fwver
    '''
    #### train
    with open(f'{data_save_path}err_type_code_{data_type}.pkl', 'rb') as f:
        err_type_code = pickle.load(f)

    if data_type == 'train':
        n_user = N_USER_TRAIN
    else:
        n_user = N_USER_TEST

    # error type과 code를 조합한 것으로 transform
    data_list = [err_type_code[user_idx] for user_idx in range(n_user)]
    err_list, model_list, fwver_list = [], [], []
    for data in tqdm(data_list):
        err_list.append(transform_errtype(data))
        model_list.append(transform_model_nm(data))
        fwver_list.append(transform_fwver(data))

    # list to array
    err_code = np.array(err_list)
    models = np.array(model_list)
    fwvers = np.array(fwver_list)

    # save
    np.save(f'{data_save_path}{data_type}_{errcode_save_name}.npy', err_code)
    np.save(f'{data_save_path}{data_type}_models.npy', models)
    np.save(f'{data_save_path}{data_type}_fwvers.npy', fwvers)

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

            t1 = number_crossing_m(train_err_r[i, :, j], 0)
            t2 = skewness(train_err_r[i, :, j])
            t3 = fourier_entropy(train_err_r[i, :, j], 1)
            t4 = cid_ce(train_err_r[i, :, j], True)
            t_nu = pd.DataFrame(data=np.array([t1, t2, t3, t4]).reshape(1, -1),
                                columns=['number_crossing_m', 'skewness', 'fourier_entropy', 'cid_ce'])

            feature_for_one_user = pd.concat([f1, f2, f3, f4, t_nu], axis=1)
            # feature_for_one_user = pd.concat([f1, f2, f3, f4], axis=1)
            feature_for_one_user.columns = [str(j) + '_' + c for c in feature_for_one_user.columns]
            tmp.append(feature_for_one_user)

        tmp = pd.concat(tmp, axis=1)
        data.append(tmp)

    features = pd.concat(data, axis=0).reset_index(drop=True)
    return features


def dimension_reduction(data, WINDOW): # 임시로는 mean 등이고library 활용할 것
    data_list = []
    for i in range(31 - WINDOW):
        sum_ = np.sum(data[:, i:i + WINDOW, :], axis=1)
        data_list.append(sum_)
    data_r = np.concatenate(
        [np.min(data_list, axis=0), np.max(data_list, axis=0), np.mean(data_list, axis=0),
         np.median(data_list, axis=0), np.std(data_list, axis=0)], axis=1)
    data_df = pd.DataFrame(data_r)
    return data_df


def extract_model_nm(data, id_list):
    model_sorted_order = np.array([4, 1, 0, 2, 8, 5, 3, 7, 6]) + 1

    # model_diff flag
    model_diff = np.zeros((len(id_list)), dtype=int)

    for i in range(len(id_list)):
        model_diff[i] = (data[i].sum(axis=0) > 0).sum() > 1

    # model_start flag
    model_start = np.zeros((len(id_list)), dtype=int)
    for i in range(len(id_list)):
        occur_idx = np.where(data[i].sum(axis=1))[0]
        if occur_idx.size == 0:
            continue
        first_occur_idx = occur_idx[0]
        model_class = np.where(data[i][first_occur_idx,:])[0][0] + 1 # 없는것과 0을 구분하기 위해 1을 더해준다
        model_start[i] = model_class

    # model_end flag
    model_end = np.zeros((len(id_list)), dtype=int)
    for i in range(len(id_list)):
        occur_idx = np.where(data[i].sum(axis=1))[0]
        if occur_idx.size == 0:
            continue
        last_occur_idx = occur_idx[-1]
        model_class = np.where(data[i][last_occur_idx,:])[0][0] + 1 # 없는것과 0을 구분하기 위해 1을 더해준다
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


def feature_extraction():
    '''
    1. Extract error type
    2. Extract model_nm
    3. Extract fwver
    4. Extract quality
    *option
    1: manually, 2: extract from library, 3. load saved data extracted from library
    '''

    train_err_arr = np.load(f'{data_save_path}train_{errcode_save_name}.npy')
    test_err_arr = np.load(f'{data_save_path}test_{errcode_save_name}.npy')

    train_err_df = dimension_reduction(train_err_arr, WINDOW=3)
    test_err_df = dimension_reduction(test_err_arr, WINDOW=3)

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

    ### 4.extract features from quality
    train_qual_arr = np.load(f'{data_save_path}train_quality.npy')
    test_qual_arr = np.load(f'{data_save_path}test_quality.npy')

    QUAL_WINDOW = 8
    # train_qual_arr = train_qual_arr[:, :, [0,2,3,4,6,9,11,12]]
    # test_qual_arr = test_qual_arr[:, :, [0,2,3,4,6,9,11,12]]
    train_quality_df = dimension_reduction(train_qual_arr, WINDOW=QUAL_WINDOW)
    test_quality_df = dimension_reduction(test_qual_arr, WINDOW=QUAL_WINDOW)

    ### End. concatenate features
    train_data = pd.concat([train_err_df, train_model_df, train_fwver_df, train_quality_df], axis=1)
    test_data = pd.concat([test_err_df, test_model_df, test_fwver_df, test_quality_df], axis=1)

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
    valid_probs = np.zeros((train_y.shape))
    # -------------------------------------------------------------------------------------
    # Kfold cross validation
    models = []
    k_fold = KFold(n_splits = 5, shuffle=True, random_state=0)
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
            # return -1, -1
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


        auc_val = model.best_score['valid_0']['auc']
        models.append(model)
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
            'learning_rate':            hp.uniform('learning_rate',    0.01, 0.02),
            'max_depth':                -1,
            'num_leaves':               hp.quniform('num_leaves',       20, 60, 1),
            'min_data_in_leaf':		    hp.quniform('min_data_in_leaf',	10, 30, 1),	# overfitting 안되려면 높은 값
            'reg_alpha':                hp.uniform('reg_alpha',0.3, 0.6),
            'reg_lambda':               hp.uniform('reg_lambda',0.6,0.8),
            'colsample_bytree':         hp.uniform('colsample_bytree', 0.1, 0.3),
            'colsample_bynode':		    hp.uniform('colsample_bynode',0.8,1.0),
            'bagging_freq':			    hp.quniform('bagging_freq',	2,10,1),
            'tree_learner':			    hp.choice('tree_learner',	['feature','data','voting']),
            'subsample':                hp.uniform('subsample', 0.9, 1.0),
            'boosting':			        hp.choice('boosting', ['gbdt']),
            'max_bin':			        hp.quniform('max_bin',		5,20,1), # overfitting 안되려면 낮은 값
            "min_sum_hessian_in_leaf":  hp.uniform('min_sum_hessian_in_leaf',       0.01,0.1),
            'random_state':             self.random_state,
            'n_jobs':                   -1,
            'metrics':                  'auc',
            'verbose':                  -1,
            'force_col_wise':           True,
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
    # new_errtype = generate_new_errtype()

    # Unknown 제거
    new_errtype_tmp = generate_new_errtype()
    new_errtype = []
    for err in new_errtype_tmp:
        if 'UNKNOWN' not in err:
            new_errtype.append(err)

    N_NEW_ERRTYPE = len(new_errtype)
    # 새로운 error type을 encoding
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(new_errtype)
    print('Encoding error code is done')
    '''
    3. 1에서 변경된 데이터를 차례로 processing
    '''
    if transform == True:
        for data_type in ['train', 'test']:
            # transform_error_data(data_type)
            transform_quality_data(data_type)
    print('Transform error and quality data is done')

    # %%+
    '''
    4. 일별 데이터에서 피쳐를 추출하여 학습에 적합한 데이터로 변경 
    *option 
    1: manually
    2: extract from library
    3. load saved data extracted from library
    '''
    # from error and quality data
    train_X, test_X = feature_extraction()

    #%% clustering
    cols = train_X.columns
    train_X.columns = range(train_X.shape[1])
    test_X.columns = range(test_X.shape[1])

    print('Feature extraction is done')

    #%%
    '''
    5. train_problem (y) 생성
    '''
    train_prob = pd.read_csv(data_path + 'train_problem_data.csv')
    train_y = np.zeros((N_USER_TRAIN))
    train_y[train_prob.user_id.unique() - 10000] = 1
    print('Make label data is done')

    '''
    6. fit
    '''
    param_select_option = input('Param: (1) default, (2) bayes opt, (3) previous best param')
    param_select_option = int(param_select_option)
    # param_select_option = 1
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
        with open(f'tune_results/0201-local-2.pkl', 'wb') as f:
            pickle.dump(bayes_trials.results, f, pickle.HIGHEST_PROTOCOL)
        print('Process 6 Done')
        params = best_param

    elif param_select_option == 3:
        from util import load_obj
        # params = load_obj('0118-2')[0]['params']
        # params = load_obj('0129-v2-local')[0]['params']
        params = load_obj('0131-local-quality-included')[0]['params']

    models, valid_probs = train_model(train_X, train_y, params)

    # evaluate
    threshold = 0.5
    valid_preds = np.where(valid_probs > threshold, 1, 0)
    recall = recall_score(train_y, valid_preds)
    precision = precision_score(train_y, valid_preds)
    auc_score = roc_auc_score(train_y, valid_probs)
     # validation score
    print('Validation score is {:.5f}'.format(auc_score))
    print('Data fitting and evaluation is done')

#%%
    '''
    7. stacking
    '''
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    blendParams = {}  # test more values in your local machine
    clf = GridSearchCV(LogisticRegression(solver='lbfgs'), blendParams,
                       scoring='roc_auc',
                       refit='True', n_jobs=-1, cv=5)
    clf.fit(valid_probs.reshape(-1, 1), train_y)
    print('The Best parameters of the blending model\n{}'.format(clf.best_params_))
    print('The best score:{}'.format(clf.best_score_))


#%%
    '''
    8. predict
    '''
    if infer == True:
        submission = pd.read_csv(data_path + '/sample_submission.csv')

        # predict
        test_prob = []
        for model in models:
            test_prob.append(model.predict(test_X))
        test_prob = np.mean(test_prob, axis=0)

        submission['problem'] = test_prob.reshape(-1)
        submission.to_csv("submission.csv", index=False)
        print('Inference is done')


#%% dumbs
    # window_err = np.array([ 6, 11, 11,  1,  1,  5,  6, 14,  1,  1,  5, 22,  0, 28,  4, 25,  0,
    #    26, 25, 25, 27,  3,  1,  0, 17, 22, 21, 13, 13,  2,  0,  0,  0, 16,
    #     3,  6,  9, 22,  6,  0, 15, 10,  0,  0,  0,  0, 13,  1,  0, 11, 17,
    #    26,  3,  1, 22,  1,  1, 14, 13,  1, 15, 17,  0, 11, 13,  3,  6, 25,
    #     1,  2,  3, 28, 27, 27, 15,  0,  0, 15, 16, 16,  0,  6,  4, 22, 10,
    #     0, 19, 13, 11,  0, 11,  0,  0,  0,  0, 26,  0,  8, 25,  5, 10, 20,
    #    11,  6, 24,  0, 11,  1,  1,  5,  3,  1,  5, 21,  0, 26, 27,  3,  1,
    #     3,  0, 24,  9, 22,  6,  0, 13,  1, 11,  1, 10,  1, 15, 15, 11, 13,
    #     6,  3,  4, 10]) + 1
    #
    # ### 0. load dataset
    # train_err_df_list, test_err_df_list = [], []
    # for j in range(train_err_arr.shape[2]):
    #     ERROR_WINDOW = window_err[j]
    #     ERROR_WINDOW = 3
    #     ### 1. extract features based on option
    #     train_err_df = dimension_reduction(train_err_arr[:,:,j].reshape(N_USER_TRAIN, 30, 1), WINDOW = ERROR_WINDOW)
    #     test_err_df = dimension_reduction(test_err_arr[:, :, j].reshape(N_USER_TEST, 30, 1), WINDOW = ERROR_WINDOW)
    #     train_err_df_list.append(train_err_df)
    #     test_err_df_list.append(test_err_df)
    # train_err_df = pd.concat(train_err_df_list, axis=1)
    # test_err_df = pd.concat(test_err_df_list, axis=1)

# window_quals = np.array([10, 5, 10, 0, 0, 22, 26, 23, 5, 10, 23, 16, 4, 9, 9, 9, 0,
    #                          0, 9, 9, 0, 0, 0, 0, 9, 0]) + 1

    # train_quality_df_list, test_quality_df_list = [], []
    # for j in range(train_qual_arr.shape[2]):
    #     QUAL_WINDOW = window_quals[j]
    #     QUAL_WINDOW = 8
    #     train_quality_df = dimension_reduction(train_qual_arr[:, :, j].reshape(N_USER_TRAIN, 31, 1),
    #                                            WINDOW=QUAL_WINDOW)
    #     test_quality_df = dimension_reduction(test_qual_arr[:, :, j].reshape(N_USER_TEST, 31, 1), WINDOW=QUAL_WINDOW)
    #     train_quality_df_list.append(train_quality_df)
    #     test_quality_df_list.append(test_quality_df)
    # train_quality_df = pd.concat(train_quality_df_list, axis=1)
    # test_quality_df = pd.concat(test_quality_df_list, axis=1)
