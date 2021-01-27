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
from tsfresh import extract_features
from tsfresh.feature_extraction.settings import EfficientFCParameters
from tsfresh.utilities.distribution import MultiprocessingDistributor

data_path = 'data/'
data_save_path = 'data_use/'
result_path = 'result_2'

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
    # train
    with Pool(NUM_CPU) as p:
        train_err_df = pd.read_csv(data_path + 'train_err_data.csv')
        train_err_df.drop_duplicates(inplace=True)
        train_err_df = train_err_df.reset_index(drop=True)
        train_err_df['time'] = pd.to_datetime(train_err_df['time'], format='%Y%m%d%H%M%S')
        process_errcode_train = partial(process_errcode, err_df=train_err_df)
        err_type_code_train = p.map(process_errcode_train, TRAIN_ID_LIST)
        del train_err_df
        # save FIXME: 제출시 save가 아니고 바로 사용하도록 바꿔야 함
        with open(f'{data_save_path}err_type_code_train.pkl', 'wb') as f:
            pickle.dump(err_type_code_train, f)
        del err_type_code_train

    # test
    with Pool(NUM_CPU) as p:
        ## test
        test_err_df = pd.read_csv(data_path + 'test_err_data.csv')
        test_err_df.drop_duplicates(inplace=True)
        test_err_df = test_err_df.reset_index(drop=True)
        test_err_df['time'] = pd.to_datetime(test_err_df['time'], format='%Y%m%d%H%M%S')
        process_errcode_test = partial(process_errcode, err_df=test_err_df)
        err_type_code_test = p.map(process_errcode_test, TEST_ID_LIST)
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
        # print(f'>>> Processing for error {e} is started...')
        idx = errortype == e

        # 값이 없을 경우
        if idx.sum() == 0:
            # print(f'<<< Processing for error {e} is done...')
            continue

        # 값이 있을 경우
        errcode = errorcode[idx]
        if e == 1:
            for i, ec in enumerate(errcode):
                if ec == '0':
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + ec
                elif ec[0] == 'P':
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'P'
                else:
                    print('Unknown error code for error type 1')
                    return -1
        elif e in [2, 4, 31, 37, 39, 40]:
            new_errtype[idx] = np.array([str(e) + '-' + ec for ec in errcode])
        elif e == 3:
            new_errtype[idx] = np.array([str(e) + '-' + ec for ec in errcode])
        elif e == 30:
            new_errtype[idx] = np.array([str(e) + '-' + ec for ec in errcode])
        elif e == 5:
            for i, ec in enumerate(errcode):
                if ec[0] in ['Y', 'V', 'U', 'S', 'Q', 'P', 'M', 'J', 'H', 'E', 'D', 'C', 'B']:
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + ec[0]
                elif ec in ['nan', 'http']:
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'num'
                else:
                    try:
                        int(ec)
                        new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'num'
                    except:
                        print('Unknown error code for error type 5: It should be int')
                        new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'num'
        elif e in [6, 7]:
            new_errtype[idx] = np.array([str(e) + '-' + ec for ec in errcode])
        elif e == 8:
            new_errtype[idx] = np.array([str(e) + '-' + ec for ec in errcode])
        elif e == 9:
            for i, ec in enumerate(errcode):
                if ec == '1':
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + ec # 1인 경우
                elif ec[0] in ['C', 'V']:
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + ec[0]
                else:
                    print('Unknown error code for error type 9')
                    return -1
        elif e in [10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 24, 26, 27, 28, 35]:
            new_errtype[idx] = np.array([str(e) + '-' + ec for ec in errcode])
        elif e == 14:
            new_errtype[idx] = np.array([str(e) + '-' + ec for ec in errcode])
        elif e == 17:
            new_errtype[idx] = np.array([str(e) + '-' + ec for ec in errcode])
        elif e == 23:
            for i, ec in enumerate(errcode):
                if 'UNKNOWN' in ec:
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'UNKNOWN'
                elif 'fail' in ec:
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'fail'
                elif 'timeout' in ec:
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'timeout'
                elif 'active' in ec:
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'active'
                elif 'standby' in ec:
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'standby'
                elif 'terminate' in ec:
                    new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'terminate'
                else:
                    print('Unknown error code for error type 23')
                    return -1
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
                else:
                    try:
                        int(ec)
                        new_errtype[np.where(idx)[0][i]] = str(e) + '-' + 'num'
                    except:
                        print('Unknown error code for error type 25: It should be int')
                        print(ec)
                        return -1
        elif e == 32:
            new_errtype[idx] = str(e) + '-num'
        elif e == 33:
            new_errtype[idx] = np.array([str(e) + '-' + ec for ec in errcode])
        elif e == 34:
            new_errtype[idx] = np.array([str(e) + '-' + ec for ec in errcode])
        elif e == 36:
            new_errtype[idx] = np.array([str(e) + '-' + ec for ec in errcode])
        elif e == 38:
            new_errtype[idx] = str(e) + '-num'
        elif e == 41:
            new_errtype[idx] = np.array([str(e) + '-' + ec for ec in errcode])
        elif e == 42:
            new_errtype[idx] = np.array([str(e) + '-' + ec for ec in errcode])
        else:
            print('Unknown error type')
            return -1
        # print(f'<<< Processing for error {e} is done...')
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
    for day in range(1, 31):
        # error가 없는 user를 skip한다
        if data == {}:
            print('Unknown user, skip it')
            break
        # error code 관련
        transformed_errcode = processing_errcode(data[day]['errtype'].values,
                                                 data[day]['errcode'].values.astype(str))

        try:
            transformed_errcode = encoder.transform(transformed_errcode)
        except ValueError or KeyError:  # 새로운 error code가 있는 경우 valid한 값만 남김
            print('Unknown error code')
            valid_errcode = []
            for i, errcode in enumerate(transformed_errcode):
                if errcode in new_errtype:
                    valid_errcode.append(errcode)
                else:
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
    return err_code, err_type


def transform_model_nm(data):
    model_nm = np.zeros((30, 9), dtype = int)
    for day in range(1, 31):
        if data == {}:
            print('Unknown user, skip it')
            break
        for model in np.unique(data[day]['model_nm']):
            model_nm[day-1, int(model[-1])] = 1
    return model_nm


def transform_error_data():
    '''
    dataframe에서 array로 변경
    1. error type and code
    2. model_nm
    '''
    #### train
    with open(f'{data_save_path}err_type_code_train.pkl', 'rb') as f:
        err_type_code_train = pickle.load(f)

    # error type과 code를 조합한 것으로 transform
    data_list = [err_type_code_train[user_idx] for user_idx in range(N_USER_TRAIN)]
    with Pool(NUM_CPU) as p:
        train_err_code_list, train_err_type_list = p.map(transform_errtype, data_list)
        model_list = p.map(transform_model_nm, data_list)

    # list to array
    train_err_code = np.array(train_err_code_list)
    train_err_type = np.array(train_err_type_list)
    train_models = np.array(model_list)

    # concatenate
    train_err_code = np.append(train_err_code, train_err_type, axis = 2)

    # save
    np.save(f'{data_save_path}train_err_code.npy', train_err_code)
    np.save(f'{data_save_path}train_models.npy', train_models)

    #### test
    with open(f'{data_save_path}err_type_code_test.pkl', 'rb') as f:
        err_type_code_test = pickle.load(f)

    # error code 관련
    # error type과 code를 조합한 것으로 transform
    data_list = [err_type_code_test[user_idx] for user_idx in range(N_USER_TEST)]
    with Pool(NUM_CPU) as p:
        test_err_code_list, test_err_type_list = p.map(transform_errtype, data_list)
        model_list = p.map(transform_model_nm, data_list)

    # list to array
    test_err_code = np.array(test_err_code_list)
    test_err_type = np.array(test_err_type_list)
    test_models = np.array(model_list)

    # concatenate and save
    test_err_code = np.append(test_err_code, test_err_type, axis=2)
    np.save(f'{data_save_path}test_err_code.npy', test_err_code)
    np.save(f'{data_save_path}test_models.npy', test_models)

    # FIXME: save 말고 바로 return으로 바꿔야 함


def extract_err_manually(data, WINDOW = 3): # 임시로는 mean 등이고library 활용할 것
    err_list = []
    for i in range(31 - WINDOW):
        sum_ = np.sum(data[:, i:i + WINDOW, :], axis=1)
        err_list.append(sum_)
    err_r = np.concatenate(
        [np.min(err_list, axis=0), np.max(err_list, axis=0), np.mean(err_list, axis=0),
         np.median(err_list, axis=0)], axis=1)
    err_df = pd.DataFrame(err_r)
    return err_df


def extract_err_library(data, WINDOW = 3): # 임시로는 mean 등이고library 활용할 것
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

    CPU_CORE = multiprocessing.cpu_count()

    Distributor = MultiprocessingDistributor(n_workers=CPU_CORE, disable_progressbar=False,
                                             progressbar_title='Feature extraction', show_warnings=False)
    tf_data = extract_features(df_list, column_id='id', distributor=Distributor,
                                default_fc_parameters=EfficientFCParameters())
    return tf_data


def extract_model_nm(data, id_list):
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

    model_df = pd.DataFrame(data = model_exist)
    model_df['model_diff'] = model_diff
    model_df['model_start'] = pd.Series(model_start, dtype='category')
    model_df['model_end'] = pd.Series(model_end, dtype='category')
    return model_df


def feature_extraction(manually = True):
    '''
    1. Extract error type
    2. Extract model_nm
    '''
    import pickle
    # FIXME : 데이터 불러오는부분 빼고 extract_err_library 함수로 만들어야 함
    with open(f'{result_path}/tf_train_1.pkl', 'rb') as f:
        tf_train_1 = pickle.load(f)
    with open(f'{result_path}/tf_train_2.pkl', 'rb') as f:
        tf_train_2 = pickle.load(f)
    with open(f'{result_path}/tf_train_3.pkl', 'rb') as f:
        tf_train_3 = pickle.load(f)
    with open(f'{result_path}/tf_test_1.pkl', 'rb') as f:
        tf_test_1 = pickle.load(f)
    with open(f'{result_path}/tf_test_2.pkl', 'rb') as f:
        tf_test_2 = pickle.load(f)
    with open(f'{result_path}/tf_test_3.pkl', 'rb') as f:
        tf_test_3 = pickle.load(f)

    tf_train = pd.concat([tf_train_1, tf_train_2, tf_train_3], axis=0).reset_index(drop=True)
    tf_test = pd.concat([tf_test_1, tf_test_2, tf_test_3], axis=0).reset_index(drop=True)
    nan_idx = np.any(pd.isnull(pd.concat([tf_train, tf_test], axis=0)), axis=0)
    tmp = pd.concat([tf_train, tf_test], axis=0)
    tf_train = tf_train.loc[:,~nan_idx]
    tf_test = tmp.iloc[N_USER_TRAIN:,:].reset_index(drop=True)
    tf_test = tf_test.loc[:,~nan_idx]

    #### train
    train_err_arr = np.load(f'{data_save_path}train_err_code.npy')
    train_err_df = extract_err_manually(train_err_arr)
    # train_err_df = extract_err_library(train_err_arr)
    train_err_df = pd.concat([tf_train, train_err_df], axis=1)

    train_models = np.load(f'{data_save_path}train_models.npy')
    train_model_df = extract_model_nm(train_models, TRAIN_ID_LIST)

    # concatenate features
    train_data = pd.concat([train_err_df, train_model_df], axis=1)
    train_data.columns = range(train_data.shape[1])

    #### test
    test_err_arr = np.load(f'{data_save_path}test_err_code.npy')
    test_err_df = extract_err_manually(test_err_arr)
    # test_err_df = extract_err_library(test_err_arr)
    test_err_df = pd.concat([tf_test, test_err_df], axis=1)

    test_models = np.load(f'{data_save_path}test_models.npy')
    test_model_df = extract_model_nm(test_models, TEST_ID_LIST)

    # concatenate features
    test_data = pd.concat([test_err_df, test_model_df], axis=1)

    test_data.columns = range(test_data.shape[1])

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
            verbose_eval=True,
        )

        # cal valid prediction
        valid_prob = model.predict(valid_x)
        valid_probs[val_idx] = valid_prob

        models.append(model)

    return models, valid_probs


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
    new_errtype = generate_new_errtype()
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

    '''
    4. 일별 데이터에서 피쳐를 추출하여 학습에 적합한 데이터로 변경 
    '''
    # from error data
    train_X, test_X = feature_extraction(manually = False)
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
    from util import load_obj
    params = load_obj('0118-3')[0]['params']
    models, valid_probs = train_model(train_X, train_y, params)

    # evaluate
    threshold = 0.5
    valid_preds = np.where(valid_probs > threshold, 1, 0)
    recall = recall_score(train_y, valid_preds)
    precision = precision_score(train_y, valid_preds)
    auc_score = roc_auc_score(train_y, valid_probs)
     # validation score
    print('Process 6 Done')
    print('Validation score is {:.5f}'.format(auc_score))

    '''
    7. predict
    '''
    submission = pd.read_csv(data_path + '/sample_submission.csv')

    # predict
    test_prob = []
    for model in models:
        test_prob.append(model.predict(test_X))
    test_prob = np.mean(test_prob, axis=0)

    submission['problem'] = test_prob.reshape(-1)
    submission.to_csv("submit/submit_10.csv", index=False)
    print('Process 7 Done')