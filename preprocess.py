# %%
import pandas as pd
import os
import numpy as np

os.listdir()

"""
데이터 설명

- 대전지역에서 측정한 실내외 19곳의 센서데이터와, 주변 지역의 기상청 공공데이터를 semi-비식별화하여 제공합니다. 

- 센서는 온도를 측정하였습니다. 

- 모든 데이터는 시간 순으로 정렬 되어 있으며 10분 단위 데이터 입니다. 

- 예측 대상(target variable)은 Y18입니다. 

train.csv 

- 30일 간의 기상청 데이터 (X00~X39) 및 센서데이터 (Y00~Y17)

- X00, X07, X28, X31, X32 기온
- X01, X06, X22, X27, X29 현지기압
- X02, X03, X18, X24, X26 풍속
- X04, X10, X21, X36, X39 일일 누적강수량
- X05, X08, X09, X23, X33 해면기압
- X11, X14, X16, X19, X34 일일 누적일사량
- X12, X20, X30, X37, X38 습도
- X13, X15, X17, X25, X35 풍향


- 이후 3일 간의 기상청 데이터 (X00~X39) 및 센서데이터 (Y18)

# Y00~Y17은 4320개만 있고 432개는 nan이다.
# 반대로 Y18은 432개만 있고 4320개는 nan이다.
# 4320은 30일 동안 10분 단위의 데이터 숫자다.

test.csv 

- train.csv 기간 이후 80일 간의 기상청 데이터 (X00~X39)

기상청 데이터 (X00~X39)를 활용하여 센서데이터 (Y00~Y18)를 각각
라벨로 활용하여 학습을 시키는 것이 목적이다.
이렇게 되면 총 데이터셋은 78192개가 된다.
"""

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 1000)
pd.set_option("display.width", 600)

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

train_df18 = train_df[4320:].dropna(axis=1)
train_df = train_df[:4320].dropna(axis=1)

basic_columns = list(train_df.columns[:41])  # id~X39
total_columns = []
for i in range(18):
    if i < 10:
        basic_columns.append(f"Y0{i}")
        total_columns.append(basic_columns.copy())
        basic_columns.pop(-1)
    else:
        basic_columns.append(f"Y{i}")
        total_columns.append(basic_columns.copy())
        basic_columns.pop(-1)

# 4320개의 데이터를 18개 모아 77760개의 데이터 생성.
# 모든 센서데이터의 column을 Y로 통일.
basic_columns.append("Y")

total_df = pd.DataFrame(data=None, columns=basic_columns)
for i in range(18):
    df = train_df[total_columns[i]]
    df.columns = basic_columns
    total_df = pd.concat([total_df, df], ignore_index=True)
# train_df18 먼저 정규화를 해놓고 나중에 추가.
train_df18.columns = basic_columns
# id,Y를 제외한 정규화.
basic_columns.pop(0)
basic_columns.pop(-1)

for x in basic_columns:
    mean, std = train_df18.agg(["mean", "std"]).loc[:, x]
    train_df18[x] = (train_df18[x] - mean) / std

# 센서데이터마다 정규화를 진행한다.
for i in range(18):
    for x in basic_columns:
        mean, std = total_df[4320 * i : 4320 * (i + 1)].agg(["mean", "std"]).loc[:, x]
        total_df[4320 * i : 4320 * (i + 1)][x] = (
            total_df[4320 * i : 4320 * (i + 1)][x] - mean
        ) / std
# 432개의 데이터를 추가해서 78192개의 데이터 생성.
total_df = pd.concat([total_df, train_df18], ignore_index=True)

for x in basic_columns:
    mean, std = test_df.agg(["mean", "std"]).loc[:, x]
    test_df[x] = (test_df[x] - mean) / std

# nan값을 모두 0으로 처리한다.
total_df = total_df.fillna(value=0)
test_df = test_df.fillna(value=0)

total_df.to_csv("data/train_preprocess.csv", index=False)
test_df.to_csv("data/test_preprocess.csv", index=False)

#%%
train = pd.read_csv("data/train_preprocess.csv")
del train["id"]

# 기상청 데이터만 추출
X_train = train.loc[:, "X00":"X39"]
Y_train = train.loc[:, "Y"]

# RNN 모델에 입력 할 수 있는 시계열 형태로 데이터 변환
# sequence 길이 5
interval = 5
encoder_list = []
decoder_list = []
target_list = []
for i in range(1, X_train.shape[0] - interval):
    encoder_list.append(np.array(X_train.iloc[i : i + interval, :-1]))
    decoder_list.append(np.array(Y_train.iloc[i : i + interval - 1]))
    target_list.append(Y_train.iloc[i + interval])

encoder_sequence = np.array(encoder_list)
decoder_sequence = np.array(decoder_list)
decoder_sequence = np.reshape(decoder_sequence, (-1, 4, 1))
target = np.array(target_list)
np.save("data/encoder_data.npy", encoder_sequence)
np.save("data/decoder_data.npy", decoder_sequence)
np.save("data/target.npy", target)
