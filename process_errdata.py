#%% 라이브러리 및 함수
import pandas as pd
import numpy as np

NUM_ERR_TYPES = 42
data_path = 'data/'
train_err = pd.read_csv(data_path + 'train_err_data.csv')
train_err['time'] = pd.to_datetime(train_err['time'])

def process_err(user_id, err_df = train_err, WINDOW = 24):
    '''
    error data를 일별 누적값으로 바꿈
    '''
    ## 1. user_id에 해당하는 data만 남김
    idx_err = err_df['user_id'] == user_id
    err_df = err_df.loc[idx_err, :]

    ## 2. WINDOW에 해당하는 날만 남김
    start_datetime = pd.to_datetime('2020-11-1')
    err_count_list = []
    while start_datetime <= pd.to_datetime('2020-12-1') - pd.Timedelta(WINDOW, unit='h'):
        end_datetime = start_datetime + pd.Timedelta(WINDOW, unit='h')  # 하루씩 lag
        err_count = np.zeros((42), dtype=int)
        if err_df.size == 0:
            pass
        else:
            idx_train_err = (err_df['time'] >= start_datetime).values & \
                            (err_df['time'] < end_datetime).values
            for i in range(1, NUM_ERR_TYPES + 1):
                # errcode 미사용
                err_count[i - 1] = (err_df['errtype'][idx_train_err] == i).sum()

        err_count_list.append(err_count)

        # update
        start_datetime += pd.Timedelta(1, unit='day')

    # Dimension reduction
    err_count_list = np.array(err_count_list)

    return err_count_list

#%% preprocessing
import multiprocessing
from multiprocessing import Pool

train_id_list = list(range(10000, 25000))

if __name__ == '__main__':
    NUM_CPU = multiprocessing.cpu_count()
    with Pool(NUM_CPU) as p:
        train_err_arr = p.map(process_err, train_id_list)

del train_err
train_err_arr = np.array(train_err_arr)