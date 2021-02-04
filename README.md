# [DACON] 시스템 품질 변화로 인한 사용자 불편 예지 AI 경진대회
* 제출일: 2020-02-07
* 팀명: asdf


## 목차
1. 라이브러리 및 데이터
2. error 데이터를 활용한 피쳐 추출


## 1. 라이브러리 및 개발 환경
* 사용 library 및 version:  

|Library|Version|
|------|------|
|pandas|1.1.4|
|numpy|1.19.2|
|scikit-learn|0.24.0|
|lightgbm|3.1.1|
|catboost|0.24.4|
|tqdm|4.54.0|

* 개발 환경: Windows 10 64비트


## 2. error 데이터를 활용한 피쳐 추출
### 2-1. 데이터 일별 분리
* error 데이터는 **time, fwver, model_nm, errtype, errcode**를 포함하고 있음
* 원본 데이터는 에러가 발생할 때마다 log가 찍혀 있는데 이를 일별로 변경하는 전처리 진행
* 해당 작업은 split_error_data_in_day 함수에서 진행되며 input과 output은 다음과 같음
```python
def split_error_data_in_day(err_df, data_type):
    """
    원본 error_data를 불러와 일별 데이터로 분리

    [Input]
    err_df: train or test error data
    data_type: train or test 지정

    [Output]
    err_df_in_day_user: user id별 일별 error data
    """
    
    # 코드 생략

    return err_df_in_day_user
``` 
* return variable인 err_df_in_day_user는 각 user의 데이터를 저장하고 있는 list
* 각각의 list는 1일~30일까지의 day index를 key로 갖고 해당하는 error dataframe을 value로 갖는 dictionary를 저장하고 있음 
* 예시로 user id 10000의 day 1이 저장하고 있는 데이터는 다음과 같음 
```ipynb
Out[4]: 
   model_nm       fwver  errtype errcode
0   model_3  05.15.2138       26       1
1   model_3  05.15.2138       15       1
2   model_3  05.15.2138       16       1
3   model_3  05.15.2138       15       1
4   model_3  05.15.2138       16       1
5   model_3  05.15.2138        4       0
6   model_3  05.15.2138        4       0
7   model_3  05.15.2138       15       1
8   model_3  05.15.2138       16       1
9   model_3  05.15.2138        4       0
10  model_3  05.15.2138        4       0
11  model_3  05.15.2138       15       1
```

### 2-2. Error type과 code
#### 2-2-0. 새로운 Error code 생성
* Error type: 1~42까지 중 29를 제외한 41개가 존재
* Error code: 각 type별로 다양하게 존재
* Error type을 비슷한 유형끼리 묶게 되면 성능이 크게 감소하는데, 반대로 type을 세분화하는 code를 활용하는 것이 유의미할 것이라 판단함
* Error code는 다음과 같은 이유로 그대로 사용할 수 없음
    1. train에는 없지만 test에 있는 error code가 다수 존재
    2. 예시로, Error type 38번에 경우 error code가 5000개 이상 존재해 개별적으로 사용할 경우 복잡도가 지나치게 커짐
* 따라서 빈도수가 높은 Error code만 남겨 type과 조합하고 그 외에는 UNKNOWN으로 처리하여 학습에 제외
* 해당 작업은 process_errcode 함수에서 진행되며 예시로 error type 1에 대한 처리는 다음과 같음
```python
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
```
* train_err_data의 error type 1의 error code 종류와 processing 결과는 다음과 같음
* 0, P로 시작하는 code, 0 이외의 숫자, UNKNOWN의 4가지 유형으로 분류하여 새로운 code 생성함

|error type|error code|processed errror code|
|------|------|------|
|1|0|1-0|
|1|P-41001|1-P|
|1|P-41007|1-P|
|1|P-41010|1-P|
|1|P-41011|1-P|
|1|100|1-num|
|1|others|1-UNKNOWN|

* train_err_data를 generate_new_errcode 함수에 입력하여 새로운 error code 조합을 생성하고 이것을 인코딩
* test_err_data는 학습 데이터의 인코더를 사용하여 변환함

#### 2-2-1. 데이터 transformation
* transform_errtype 함수를 이용해 각 user, day마다의 error dataframe을 error array 형태로 변경
* 이 때, error type과 code는 일별 발생 횟수를 집계함
* 예외로 error type 38의 error code는 numeric value라 판단하여 발생 횟수가 아닌 code 숫자의 합계에 대한 피쳐 생성
* error type (42가지) + error code (x가지) + error 38 관련 피쳐 1가지로, (user id 수) x (30) x (x) 형태의 array 반환

#### 2-2-2. Feature extraction
* extract_err 함수를 통해 피쳐를 생성함
    1. 일별 데이터를 아래 그림과 같이 WINDOW마다 summation 
![img1](img/img1.png)
    2. 각 error에 대하여 차원 축소를 위해 5가지의 통계적 특징 추출 (mean, max, min, median, standard deviavion)
* extract_err 함수는 (userid 수) x (error type 수 + error code 수 + 1) 의 형태를 가진 dataframe을 반환함

### 2-3. Model_nm
#### 2-2-1. 데이터 transformation
* transform_model_nm 함수를 이용해 각 user, day마다 error dataframe을 model_nm array 형태로 변경
* 이 때, 요일별로 어떤 model_nm이었는지 저장

#### 2-2-2. Feature extraction
* extract_model_nm 함수를 통해 피쳐를 생성하였고 각각에 대한 설명은 다음과 같음

|Feature|Explanation|
|------|------|
|model_exist|0~8 model_nm 사용 유무에 대한 flag|
|model_diff|30일 내에 유저의 model_nm이 변경되었는지에 대한 flag|
|model_upgrade|model_nm 업그레이드 유무에 대한 flag|
|model_downgrade|model_nm 다운그레이드 유무에 대한 flag|
|model_start|유저가 가장 처음에 사용한 model_nm에 대한 범주형 변수|
|model_end|유저가 가장 마지막 사용한 model_nm에 대한 범주형 변수|

* upgrade 및 downgrade는 model 번호가 아닌 fwver을 기준으로 판단하였으며 그에 따른 순서는 아래 표와 같음

|model_nm|fwver|order|
|------|------|------|
|0|04.22.x|3|
|1|04.16.x|2|
|2|04.33.x|4|
|3|05.15.x|7|
|4|03.11.x|1|
|5|04.82.x|6|
|6|10 / 8.5.3|9|
|7|05.66.x|8|
|8|04.73.x|5|

### 2-4. fwver
#### 2-2-1. 데이터 transformation
* transform_fwver 함수를 이용해 각 user, day마다 error dataframe을 fwver array 형태로 변경
* 이 때, 요일별로 어떤 fwver였는지 저장
* fwver이 x. y. z의 형태이므로 이 3가지를 분리하여 저장

#### 2-2-2. Feature extraction
* extract_fwver 함수를 통해 피쳐를 생성하였고 각각에 대한 설명은 다음과 같음
* 피쳐의 특성은 model_nm과 유사함
* fwver의 x. y. z 중 x만 사용하였으며 나머지 자리는 유의미한 모델 성능 향상을 보이지 않아 제외함

|Feature|Explanation|
|------|------|
|fwver_diff|30일 내에 유저의 fwver이 변경되었는지에 대한 flag|
|fwver_upgrade|fwver 업그레이드 유무에 대한 flag|
|fwver_downgrade|fwver 다운그레이드 유무에 대한 flag|
|fwver_start|유저가 가장 처음에 사용한 fwver에 대한 범주형 변수|
|fwver_end|유저가 가장 마지막 사용한 fwver에 대한 범주형 변수|

### 2-5. 피쳐 통합
* 2-1 ~ 2-4에서 생성된 피쳐의 통합은