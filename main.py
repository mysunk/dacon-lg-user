# %% 라이브러리 임포트 변수 정의
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import *
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

# 데이터 경로 설정
data_path = 'data/'

# global 변수 정의
TRAIN_ID_LIST = list(range(10000, 25000))
TEST_ID_LIST = list(range(30000, 45000 - 1))
N_USER_TRAIN = len(TRAIN_ID_LIST)
N_USER_TEST = len(TEST_ID_LIST)
N_ERRTYPE = 42
N_QUALITY = 13


# %% error type과 code를 조합하여 새로운 error type 생성
from sklearn.preprocessing import LabelEncoder

def process_errcode(errortype, errorcode):
    """
    errortype과 errorcode를 조합해서 새로운 error type을 생성
    ex: error type이 8이고 error code가 3이면 새로운 error code는 8-3으로 변경

    [ Input ]
    errortype: error type에 해당하는 array
    errorcode: 해당 error type의 error code array

    [ Output ]
    new_errcode: error type과 code를 조합하여 생성된 array
    """

    new_errcode = errortype.copy().astype(str)
    for e in range(1, 43):
        idx = errortype == e

        # -------------------------------- 값이 없을 경우 continue
        if idx.sum() == 0:
            continue

        # -------------------------------- 값이 있을 경우에 processing
        errcode = errorcode[idx]

        # 공백 및 문자 전처리
        for i, ec in enumerate(errcode.copy()):
            ec = ec.strip()  # 앞뒤로 공백 제거
            ec = ec.replace('_', '-')
            if '.' in ec:
                ec = ec.split('.')[0]
            if ec.isdigit():
                errcode[i] = str(int(ec))
            errcode[i] = ec

        # error type에 따라 processing
        if e == 1:
            for i, ec in enumerate(errcode):
                if ec == '0':
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-' + ec
                elif ec[0] == 'P':
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-' + 'P'
                elif ec.isdigit():
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-' + 'num'
                else:
                    print(f'Unknown error code for error type {e}')
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-' + 'UNKNOWN'
        elif e == 5:
            for i, ec in enumerate(errcode):
                if ec[0] in ['Y', 'V', 'U', 'S', 'Q', 'P', 'M', 'J', 'H', 'E', 'D', 'C', 'B']:
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-' + ec[0]
                elif ec == 'nan':
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-' + 'UNKNOWN'
                elif ec.isdigit():
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-' + 'num'
                else:
                    print(f'UNKNOWN error code for type {e} :: {ec}')
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-' + 'UNKNOWN'
        elif e == 8:
            for i, ec in enumerate(errcode):
                if ec.isdigit():
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-' + 'num'
                elif ec in ['PHONE-ERR', 'PUBLIC-ERR']:
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-' + ec
                else:
                    print(f'UNKNOWN error code for type {e} :: {ec}')
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-' + 'UNKNOWN'
        elif e == 9:
            for i, ec in enumerate(errcode):
                if ec.isdigit():
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-num'
                elif ec[0] in ['C', 'V']:
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-' + ec[0]
                else:
                    print(f'Unknown error code for error type {e}')
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-UNKNOWN'
        elif e == 25:
            for i, ec in enumerate(errcode):
                if 'UNKNOWN' in ec:
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-' + 'UNKNOWN'
                elif 'fail' in ec:
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-' + 'fail'
                elif 'timeout' in ec:
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-' + 'timeout'
                elif 'cancel' in ec:
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-' + 'cancel'
                elif 'terminate' in ec:
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-' + 'terminate'
                elif ec.isdigit():
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-' + 'num'
                else:
                    print(f'Unknown error code for error type {e}')
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-' + 'UNKNOWN'
        elif e == 32:
            for i, ec in enumerate(errcode):
                if ec[0] == '-':
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-' + 'neg_num'
                else:
                    new_errcode[np.where(idx)[0][i]] = str(e) + '-' + 'pos_num'
        elif e == 38:
            new_errcode[idx] = str(e) + '-num'
        else:
            # 나머지 error type은 개별 error code를 전부 사용
            new_errcode[idx] = np.array([str(e) + '-' + ec for ec in errcode])

    return new_errcode


def generate_new_errcode():
    """
    train error data에서 process_errcode 함수를 이용하여 새로운 error type을 encoding

    [Input]
    없음

    [Output]
    new_errcode: 인코딩된 새로운 error type
    """

    err_df = pd.read_csv(data_path + 'train_err_data.csv')
    err_df = err_df.loc[:, 'errtype':'errcode']

    # make unique pairs
    err_df.drop_duplicates(inplace=True)
    err_df = err_df.reset_index(drop=True)

    # dataframe을 array로 변경
    errortype = err_df['errtype'].values
    errorcode = err_df['errcode'].values.astype(str)

    # process_errcode 함수를 이용해 새로운 error type을 encoding
    new_errcode = process_errcode(errortype, errorcode)
    new_errcode = np.unique(new_errcode)

    return new_errcode

new_errcode_tmp = generate_new_errcode()

new_errcode = []
# Unknown 제거
for err in new_errcode_tmp:
    if 'UNKNOWN' not in err:
        new_errcode.append(err)
N_NEW_ERRCODE = len(new_errcode)

# 새로운 error type을 encoding
encoder = LabelEncoder()
encoder.fit(new_errcode)
print('Encoding error code is done')


# %% X, y 생성

def split_error_data_in_day(data_type):
    """
    원본 error_data를 불러와 일별 데이터로 분리

    [Input]
    data_type: train or test 지정

    [Output]
    err_df_in_day_user: user id별 일별 error data
    """

    if data_type == 'train':
        user_id_list = TRAIN_ID_LIST
    elif data_type == 'test':
        user_id_list = TEST_ID_LIST

    # 데이터 로드
    err_df = pd.read_csv(data_path + f'{data_type}_err_data.csv')

    # 중복 제거
    err_df.drop_duplicates(inplace=True)
    err_df = err_df.reset_index(drop=True)

    # time을 datetime format으로 변경
    err_df['time'] = pd.to_datetime(err_df['time'], format='%Y%m%d%H%M%S')

    # user id 각각에 대해 processing
    err_df_in_day_user = []
    for user_id in tqdm(user_id_list):
        # error data중 user id가 user_id인 데이터
        idx = err_df['user_id'] == user_id
        err_df_sub = err_df.loc[idx, :]

        # key가 day (1~30) 이고 value가 해당 err_df에서 user와 day에 해당하는 부분인 dictionary 정의
        err_in_day = dict()

        # -------------------------------- 해당 user가 error data가 없을 시
        if idx.sum() == 0:
            err_df_in_day_user.append(err_in_day)
            continue

        # -------------------------------- user의 error data가 있을 시 1일부터 31일에 대한 처리
        for day in range(1, 31):
            idx_2 = err_df_sub['time'].dt.day == day
            err_in_day[day] = err_df_sub.loc[idx_2, 'model_nm': 'errcode'].reset_index(drop=True)

        err_df_in_day_user.append(err_in_day)
    del err_df

    return err_df_in_day_user


def transform_errtype(user_data):
    """
    일별 error dataframe을 이용하여 error를 array 형태로 변경

    [Input]
    user_data: 한 user에 대해 1일 ~ 31일까지로 분리된 error data dataframe

    [Output]
    err: error code, type, errtype_38_errcode_sum이 합쳐진 파생 array
    """

    err_code = np.zeros((30, N_NEW_ERRCODE))
    err_type = np.zeros((30, N_ERRTYPE))
    errtype_38_errcode_sum = np.zeros((30, 1), dtype=int)

    # 1일~31까지에 대한 처리
    for day in range(1, 31):

        # error가 없는 user를 skip
        if user_data == {}:
            print('Unknown user, skip it')
            break

        # error type과 code를 조합하여 새로운 error code 생성
        transformed_errcode = process_errcode(user_data[day]['errtype'].values,
                                              user_data[day]['errcode'].values.astype(str))

        # 생성된 error code를 encoding
        try:
            transformed_errcode = encoder.transform(transformed_errcode)
        # 새로운 error code가 있는 경우 valid한 값만 남김
        except ValueError or KeyError:
            valid_errcode = []
            for i, errcode in enumerate(transformed_errcode):
                if errcode in new_errcode:
                    valid_errcode.append(errcode)
                else:
                    # errcode에 'UNKNOWN'이 포함되어 있을 시 제외
                    if 'UNKNOWN' not in errcode:
                        print(f'Skip error code {errcode}')

            # valid한 error code만 transform
            transformed_errcode = encoder.transform(valid_errcode)

        # count error code
        v, c = np.unique(transformed_errcode, return_counts=True)
        if v.size == 0:
            continue
        err_code[day - 1][v] += c

        # error type 38만 따로 처리
        idx_38 = user_data[day]['errtype'].values == 38
        errtype_38_errcode_sum[day - 1] = np.sum(user_data[day]['errcode'].values[idx_38].astype(int))

        # cout error type
        errtype = user_data[day][
                      'errtype'].values - 1  # error type이 1~42이므로 index로 바꾸기 위해 1을 빼줌
        v, c = np.unique(errtype, return_counts=True)
        if v.size == 0:
            continue
        err_type[day - 1][v] += c

    # error code, type, 38 관련 피쳐 concatenate
    err = np.concatenate([err_code, err_type, errtype_38_errcode_sum], axis=1)

    return err


def transform_model_nm(data):
    """
    일별 error dataframe을 이용하여 model_nm을 array형태로 변경

    [Input]
    data: 1일 ~ 31일까지로 분리된 error data dataframe

    [Output]
    model_nm: data에서 model_nm정보만 array형태로 변환
    """

    model_nm = np.zeros((30, 9), dtype=int)
    for day in range(1, 31):
        # 데이터가 없을 시 break
        if data == {}:
            print('Unknown user, skip it')
            break

        # 해당 day에 어떤 model_nm이었는지 저장
        for model in np.unique(data[day]['model_nm']):
            model_nm[day - 1, int(model[-1])] = 1

    return model_nm


def transform_fwver(data):
    """
    일별 error dataframe을 이용하여 model_nm을 array형태로 변경

    [Input]
    data: 1일 ~ 31일까지로 분리된 error data dataframe

    [Output]
    fwver: data에서 fwver정보만 array형태로 변환
    """

    fwver = np.zeros((30, 3), dtype=int)  # 00. 00. 00로 분류
    fwver = fwver.astype(str)

    for day in range(1, 31):
        # 데이터가 없을 시 break
        if data == {}:
            print('Unknown user, skip it')
            break

        # 해당 day에 어떤 model_nm이었는지 저장
        for fwver_u in np.unique(data[day]['fwver']):
            striped_fwver = fwver_u.split('.')
            for i in range(len(striped_fwver)):
                fwver[day - 1, i] = striped_fwver[i]

    return fwver


def extract_err(err_arr, WINDOW):
    """
    error 30일 데이터를 WINDOW마다 summation하고 하루씩 lag # FIXME

    [Input]
    err_arr: (user id 수) x (day 수 = 30) x (error 수 = N_NEW_ERRCODE + N_ERRTYPE)의 형태를 가진 error array
    WINDOW: window length

    [OUTPUT]
    err_df: features에 해당하는 통계적 특징을 추출한 dataframe
    """

    # 하루씩 lag하며 WINDOW에 해당하는 날의 error count를 summation
    err_list = []
    for i in range(31 - WINDOW):
        sum_ = np.sum(err_arr[:, i:i + WINDOW, :], axis=1)
        err_list.append(sum_)
    err_r = np.concatenate(
        [np.min(err_list, axis=0), np.max(err_list, axis=0), np.mean(err_list, axis=0),
         np.median(err_list, axis=0), np.std(err_list, axis=0)], axis=1)
    err_df = pd.DataFrame(err_r)

    # dataframe의 가독성을 위한 column명 지정
    features = ['min', 'max', 'mean', 'median', 'std']
    data_cols = []
    for feature in features:
        for i in range(N_NEW_ERRCODE):
            data_cols.append('errorcode_' + feature + '_' + str(i))
        for i in range(N_ERRTYPE):
            data_cols.append('errortype_' + feature + '_' + str(i))
        data_cols.append('errortype_38_code_summation_' + feature)
    err_df.columns = data_cols

    return err_df


def extract_model_nm(model_arr, id_list):
    """
    model array에서 통계적 특징을 추출

    [Input]
    model_arr: (user id 수) x (day 수 = 30) x (model 수 = 9)의 형태를 가진 model array
    id_list: train or test id list

    [OUTPUT]
    model_df: model_nm에서 추출된 특징을 조합한 dataframe
    """

    # train 데이터의 model_nm과 fwver를 기반으로 model의 순서를 정의
    model_sorted_order = np.array([4, 1, 0, 2, 8, 5, 3, 7, 6]) + 1

    # model_diff flag
    model_diff = np.zeros((len(id_list)), dtype=int)

    for i in range(len(id_list)):
        model_diff[i] = (model_arr[i].sum(axis=0) > 0).sum() > 1

    # model_start flag
    model_start = np.zeros((len(id_list)), dtype=int)
    for i in range(len(id_list)):
        occur_idx = np.where(model_arr[i].sum(axis=1))[0]
        if occur_idx.size == 0:
            continue
        first_occur_idx = occur_idx[0]
        model_class = np.where(model_arr[i][first_occur_idx, :])[0][0] + 1  # 없는것과 0을 구분하기 위해 1을 더해준다
        model_start[i] = model_class

    # model_end flag
    model_end = np.zeros((len(id_list)), dtype=int)
    for i in range(len(id_list)):
        occur_idx = np.where(model_arr[i].sum(axis=1))[0]
        if occur_idx.size == 0:
            continue
        last_occur_idx = occur_idx[-1]
        model_class = np.where(model_arr[i][last_occur_idx, :])[0][0] + 1  # 없는것과 0을 구분하기 위해 1을 더해준다
        model_end[i] = model_class

    # model 유무
    model_exist = np.zeros((len(id_list), 9), dtype=int)
    for i in range(len(id_list)):
        model_exist[i, :] = model_arr[i].sum(axis=0) > 0

    # model upgrade
    model_upgrade = np.zeros((len(id_list)), dtype=int)
    model_downgrade = np.zeros((len(id_list)), dtype=int)
    for i in range(len(id_list)):
        if model_start[i] == 0 or model_end[i] == 0:  # unknown user
            continue
        idx1 = np.where(model_start[i] == model_sorted_order)[0][0]
        idx2 = np.where(model_end[i] == model_sorted_order)[0][0]
        if idx1 == idx2:
            pass
        elif idx1 < idx2:
            model_upgrade[i] = 1
        else:
            model_downgrade[i] = -1

    # 처리된 데이터 통합
    model_df = pd.DataFrame(data=model_exist, columns=['model_' + str(i) for i in range(9)])
    model_df['model_diff'] = model_diff
    model_df['model_start'] = pd.Series(model_start, dtype='category')
    model_df['model_end'] = pd.Series(model_end, dtype='category')
    model_df['model_upgrade'] = model_upgrade
    model_df['model_downgrade'] = model_downgrade

    return model_df


def extract_fwver(fwver_arr, id_list):
    """
    model array에서 통계적 특징을 추출

    [Input]
    model_arr: (user id 수) x (day 수 = 30) x (model 수 = 9)의 형태를 가진 model array
    id_list: train or test id list

    [OUTPUT]
    model_df: model_nm에서 추출된 특징을 조합한 dataframe
    """

    # int type으로 변경
    fwver_arr = fwver_arr.astype(int)

    fwver_diff = np.zeros((len(id_list), 3), dtype=int)
    fwver_start = np.zeros((len(id_list), 3), dtype=int)
    fwver_end = np.zeros((len(id_list), 3), dtype=int)
    fwver_upgrade = np.zeros((len(id_list), 3), dtype=int)
    fwver_downgrade = np.zeros((len(id_list), 3), dtype=int)

    # 각 user마다
    for i in range(len(id_list)):
        for j in range(3):
            data_sub = fwver_arr[i][:, j]
            data_sub = data_sub[data_sub != 0]

            # fwver diff
            fwver_diff[i, j] = np.max([len(np.unique(data_sub)) - 1, 0])
            occur_idx = np.where(data_sub)[0]
            if occur_idx.size == 0:
                continue

            # fwver start
            first_occur_idx = occur_idx[0]
            fwver_class_start = data_sub[first_occur_idx]
            fwver_start[i, j] = fwver_class_start

            # fwver end
            last_occur_idx = occur_idx[-1]
            fwver_class_end = data_sub[last_occur_idx]
            fwver_end[i, j] = fwver_class_end

            # fwver upgrade or downgrade
            fwver_upgrade[i, j] = fwver_class_start < fwver_class_end
            fwver_downgrade[i, j] = (fwver_class_start > fwver_class_end) * -1

    fwver_df = pd.DataFrame()
    for j in range(3):
        fwver_df['fwver_diff_' + str(j)] = fwver_diff[:, j]
        fwver_df['fwver_start_' + str(j)] = pd.Series(fwver_start[:, j])
        fwver_df['fwver_end_' + str(j)] = pd.Series(fwver_end[:, j])
        fwver_df['fwver_upgrade_' + str(j)] = pd.Series(fwver_upgrade[:, j])
        fwver_df['fwver_downgrade_' + str(j)] = pd.Series(fwver_downgrade[:, j])
        break # 가장 앞자리만 사용
    return fwver_df


def transform_extract_quality_data(data_type):
    '''
    quality data에서 feature extraction

    [Input]
    data_type: train or test 지정

    [Output]
    qual_features: quality 데이터의 통계적 특징을 반영한 데이터
    '''

    if data_type == 'train':
        user_id_list = TRAIN_ID_LIST
    else:
        user_id_list = TEST_ID_LIST

    # 데이터 로드
    quality = pd.read_csv(data_path + f'{data_type}_quality_data.csv')

    # time을 datetime으로 변경
    quality['time'] = pd.to_datetime(quality['time'], format='%Y%m%d%H%M%S')

    # 숫자에 포함된 ','를 제거
    quality.replace(",", "", regex=True, inplace=True)

    # quality data의 type을 float으로 변경
    quality.loc[:, 'quality_0':'quality_12'] = quality.loc[:, 'quality_0':'quality_12'].astype(float)

    # drop nan
    drop_idx = np.any(pd.isnull(quality), axis=1)
    quality = quality.loc[~drop_idx, :].reset_index(drop=True)

    # user id별로 그루핑
    group = quality.groupby('user_id')
    qual_features = pd.DataFrame(index=user_id_list)
    qual_user_id_list = np.unique(quality['user_id'])

    # 각 user id에 대하여 processing
    for user_id in tqdm(qual_user_id_list):
        qual = group.get_group(user_id)
        qual = qual.loc[:, 'quality_0':'quality_12']

        # -1을 count해서 저장
        minus_1_idx = qual == -1
        for i in [1, 2, 5, 6, 7, 8, 9, 10, 11, 12]:
            qual_features.loc[user_id, f'quality_{i}_minus_1_counts'] = minus_1_idx.sum(axis=0).loc[f'quality_{i}']

        # drop minus 1
        minus_1_idx_row = np.any(minus_1_idx, axis=1)
        qual = qual.loc[~minus_1_idx_row, :]

        # feature extraction:: mean, max, median, std
        for i in [1, 2, 5, 6, 7, 8, 9, 10, 11, 12]:
            qual_features.loc[user_id, f'quality_{i}_mean'] = qual.loc[:, f'quality_{i}'].mean(axis=0)
            qual_features.loc[user_id, f'quality_{i}_max'] = qual.loc[:, f'quality_{i}'].max(axis=0)
            qual_features.loc[user_id, f'quality_{i}_med'] = qual.loc[:, f'quality_{i}'].median(axis=0)
            qual_features.loc[user_id, f'quality_{i}_std'] = np.std(qual.loc[:, f'quality_{i}'], axis=0)

    # nan을 0으로 변경
    qual_features = qual_features.reset_index(drop=True)
    qual_features = qual_features.fillna(0)

    return qual_features


def feature_extraction(data_type):
    """
    다음 순서로 피쳐를 추출
    1. Extract error type, code
    2. Extract model_nm
    3. Extract fwver
    4. Extract quality

    [Input]
    data_type: train or test

    [Output]
    transformed_data: error, model, fwver, quality에서 추출된 피쳐를 합친 데이터
    """
    if data_type == 'train':
        n_user = N_USER_TRAIN
        id_list = TRAIN_ID_LIST
    elif data_type == 'test':
        n_user = N_USER_TEST
        id_list = TEST_ID_LIST

    # -------------------------------- error 데이터 처리
    # raw error 데이터를 일별, 유저별로 분리
    print('Splitting error data in day...')
    err_df_in_day_user = split_error_data_in_day(data_type)
    data_list = []
    for user_idx in range(n_user):
        data_list.append(err_df_in_day_user[user_idx])
    print('Splitting error data in day is done')

    ### 1. extract features from error
    print('Extract features from error...')
    err_list = []
    for data in tqdm(data_list):
        err_list.append(transform_errtype(data))
    err_arr = np.array(err_list)
    err_df = extract_err(err_arr, WINDOW=3)
    print('Extract features from error is done')

    ### 2. extract features from model_nm
    print('Extract features from model...')
    model_list = []
    for data in tqdm(data_list):
        model_list.append(transform_model_nm(data))
    model_arr = np.array(model_list)
    model_df = extract_model_nm(model_arr, id_list)
    print('Extract features from model is done')

    ### 3. extract features from fwver
    print('Extract features from fwver...')
    fwver_list = []
    for data in tqdm(data_list):
        fwver_list.append(transform_fwver(data))
    fwver_arr = np.array(fwver_list)
    fwver_df = extract_fwver(fwver_arr, id_list)
    print('Extract features from fwver is done')

    # -------------------------------- quality 데이터 처리
    ### 4. quality data transformation and feature extraction
    print('Extract features from quality...')
    quality_df = transform_extract_quality_data(data_type)
    print('Extract features from quality is done')

    # -------------------------------- Concatenate features
    transformed_data = pd.concat([err_df, model_df, fwver_df, quality_df], axis=1)

    return transformed_data


# X 생성
print('Feature extraction from train data start')
train_X = feature_extraction('train')
print('Feature extraction from train data end')

print('Feature extraction from test data start')
test_X = feature_extraction('test')
print('Feature extraction from test data ent')

# y 생성
train_prob = pd.read_csv(data_path + 'train_problem_data.csv')
train_y = np.zeros((N_USER_TRAIN))
train_y[train_prob.user_id.unique() - 10000] = 1
print('Make train, test and label data is done')


# %% 모델 학습
def f_pr_auc(probas_pred, y_true):
    """
    lightgbm custom loss functions

    [Input]
    probas_pred: 예측 결과
    y_true: 실제값

    [Output]
    metric 이름, auc 값, whether maximize or minimize (True is maximize)
    """

    labels = y_true.get_label()
    p, r, _ = precision_recall_curve(labels, probas_pred)
    score = auc(r, p)
    return "pr_auc", score, True


def lgb_train_model(train_x, train_y, params, n_fold, fold_rs=0):
    """
    lightgbm cross validation with given data

    [Input]
    train_x, train_y: 학습에 사용될 data와 label
    params: lightgbm parameter
    n_fold: k-fold cross validation의 fold 수
    fold_rs: fold를 나눌 random state

    [Output]
    models: fold별로 피팅된 model
    valid_probs: validation set에 대한 예측 결과
    """

    valid_probs = np.zeros((train_y.shape))

    # -------------------------------------------------------------------------------------
    # Kfold cross validation

    models = []
    k_fold = KFold(n_splits=n_fold, shuffle=True, random_state=fold_rs)
    # split train, validation set
    for train_idx, val_idx in k_fold.split(train_x):

        # input 데이터 형식이 dataframe일 때와 array일 때를 구분
        if type(train_x) == pd.DataFrame:
            X = train_x.iloc[train_idx, :]
            valid_x = train_x.iloc[val_idx, :]
        elif type(train_x) == np.ndarray:
            X = train_x[train_idx, :]
            valid_x = train_x[val_idx, :]
        else:
            print('Unknown data type for X')
            return -1, -1

        y = train_y[train_idx]
        valid_y = train_y[val_idx]

        d_train = lgb.Dataset(X, y, categorical_feature=['model_start','model_end'])
        d_val = lgb.Dataset(valid_x, valid_y, categorical_feature=['model_start','model_end'])

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

    return models, valid_probs


def cat_train_model(train_x, train_y, params, n_fold, fold_rs=0):
    """
    catboost cross validation with given data

    [Input]
    train_x, train_y: 학습에 사용될 data와 label
    params: catboost parameter
    n_fold: k-fold cross validation의 fold 수
    fold_rs: fold를 나눌 random state

    [Output]
    models: fold별로 피팅된 model
    valid_probs: validation set에 대한 예측 결과
    """

    valid_probs = np.zeros((train_y.shape))
    models = []
    # -------------------------------------------------------------------------------------
    # Kfold cross validation
    k_fold = KFold(n_splits=n_fold, shuffle=True, random_state=fold_rs)
    for train_idx, val_idx in k_fold.split(train_x):
        # split train, validation set
        if type(train_x) == pd.DataFrame:
            X = train_x.iloc[train_idx, :]
            valid_x = train_x.iloc[val_idx, :]
        elif type(train_x) == np.ndarray:
            X = train_x[train_idx, :]
            valid_x = train_x[val_idx, :]
        else:
            print('Unknown data type for X')
            # return -1, -1
        y = train_y[train_idx]
        valid_y = train_y[val_idx]

        train_dataset = Pool(data=X,
                             label=y,
                             cat_features=['model_start', 'model_end'])

        valid_dataset = Pool(data=valid_x,
                             label=valid_y,
                             cat_features=['model_start', 'model_end'])

        cbm_clf = CatBoostClassifier(**params)

        cbm_clf.fit(train_dataset,
                    eval_set=valid_dataset,
                    verbose=False,
                    plot=False,
                    )

        # cal valid prediction
        valid_prob = cbm_clf.predict_proba(valid_x)
        valid_probs[val_idx] = valid_prob[:, 1]

        models.append(cbm_clf)

    return models, valid_probs


lgb_param = {
    'force_col_wise': True,
    'objective': 'binary',
    'tree_learner': 'data',
    'boosting': 'gbdt',
    'metrics': 'auc',
    'colsample_bynode': 0.3898135486693247,
    'colsample_bytree': 0.18842555877544384,
    'learning_rate': 0.017650166362369914,
    'min_sum_hessian_in_leaf': 0.06737426241310715,
    'reg_alpha': 0.8971653884407311,
    'reg_lambda': 0.11885254289919682,
    'subsample': 0.960014724793571,
    'random_state': 0,
    'verbose': -1,
    'max_depth': -1,
    'bagging_freq': 19,
    'min_data_in_leaf': 25,
    'max_bin': 5,
    'n_jobs': 6,
    'num_leaves': 40,
}

cat_param = {
    'custom_loss': 'AUC',
    'bagging_temperature': 0.0002815749995043748,
    'learning_rate': 0.07949149044006878,
    'random_strength': 1326.5863036592555,
    'scale_pos_weight': 0.563214416647416,
    'subsample': 0.6014021044879473,
    'colsample_bylevel': 0.46663684733988225,
    'border_count': 59,
    'depth': 7,
    'iterations': 816,
    'l2_leaf_reg': 47,
    'thread_count': 48
}

valid_probs_lgb_list = []
for i in range(10):
    # k-fold cross validation의 random state를 바꿔가며 앙상블
    models_lgb, valid_probs_lgb = lgb_train_model(train_X, train_y, lgb_param, n_fold = 10, fold_rs=i)
    valid_probs_lgb_list.append(valid_probs_lgb)
auc_score_lgb = roc_auc_score(train_y, np.mean(valid_probs_lgb_list, axis=0))
print('Lightgbm validation score:: {:.5f}'.format(auc_score_lgb))
valid_probs_lgb = np.mean(valid_probs_lgb_list, axis=0)

print('Catboost model fitting...')
valid_probs_cat_list = []
for i in range(10):
    # k-fold cross validation의 random state를 바꿔가며 앙상블
    models_cat, valid_probs_cat = cat_train_model(train_X, train_y, cat_param, n_fold = 10, fold_rs=i)
    valid_probs_cat_list.append(valid_probs_cat)
auc_score_cat = roc_auc_score(train_y, np.mean(valid_probs_cat_list, axis=0))
print('Catboost validation score:: {:.5f}'.format(auc_score_cat))
valid_probs_cat = np.mean(valid_probs_cat_list, axis=0)

# lightgbm 모델과 catboost 모델을 앙상블
# 앙상블 weight 조정
auc_score_ens_list = []
for i in range(0, 11):
    auc_score_ens = roc_auc_score(train_y, valid_probs_lgb * (0.1 * i) + valid_probs_cat * (1 - (0.1 * i)))
    auc_score_ens_list.append(auc_score_ens)
print('Ensemble validation score:: {:.5f}'.format(np.max(auc_score_ens_list)))

i = np.argmax(auc_score_ens_list)
weight = [0.1 * i, (1 - (0.1 * i))]

# %% Inference
print('Inference for test dataset...')

submission = pd.read_csv(data_path + '/sample_submission.csv')

# lightgbm prediction
test_prob_lgb = []
for model in models_lgb:
    test_prob_lgb.append(model.predict(test_X))
test_prob_lgb = np.mean(test_prob_lgb, axis=0)

# catboost prediction
test_prob_cat = []
for model in models_cat:
    test_prob_cat.append(model.predict(test_X))
test_prob_cat = np.mean(test_prob_cat, axis=0)

test_prob = test_prob_lgb * weight[0] + test_prob_cat * weight[1]

submission['problem'] = test_prob.reshape(-1)
submission.to_csv("submission.csv", index=False)

print('Inference is done')