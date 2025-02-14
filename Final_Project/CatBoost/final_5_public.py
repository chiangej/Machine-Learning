import pandas as pd
import data_process
import numpy as np
from keras.callbacks import  EarlyStopping
from keras.models import Sequential
from keras.src.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
station = []
with open(r"C:\Users\chiangejPycharmProjects\ML_Final_Project\html.2023.final.data\sno_test_set.txt", 'r', encoding='utf-8') as file:
    lines = file.readlines()
    station.append(lines)
station = [item.strip() for sublist in station for item in sublist]

for j in range(0, 39):
    station_id = station[j]

    dfs = []
    result = pd.DataFrame()
    #取10月資料(從2號開始)
    for i in range(1, 11):
        day = i + 1
        if i <= 8:
            one_day = data_process.read_data(f"2023100{day}", station_id)
        else:
            one_day = data_process.read_data(f"202310{day}", station_id)
        dfs.append(one_day)

    for i in range(6):
        day = i + 15
        one_day = data_process.read_data(f"202310{day}", station_id)
        dfs.append(one_day)

    result = pd.concat(dfs, ignore_index=True)
    big, small = result["sbi"].max(), result["sbi"].min()
    data_nor = data_process.normalize(result)

    data_num = 289
    x_train = []
    y_train = []
    print(data_num)
    for i in range(data_num):
        subset_df = data_nor.head(576)
        x_train.append(subset_df.head(288).values.tolist())
        y_train.append(subset_df.tail(288)["sbi"].values.tolist())
        data_nor = data_nor.iloc[2:]

    x_test = data_nor.tail(288).values.tolist()
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (1, 288, 6))
    y_train = np.reshape(y_train, (289, 288))
    print(x_train.shape)
    print(y_train)
    print(x_test)


    model = Sequential()
    model.add(LSTM(256, input_shape=(x_train.shape[1], 6), return_sequences=True))
    model.add(Dropout(0.01))
    model.add(LSTM(256, input_shape=(x_train.shape[1], 6), return_sequences=True))
    model.add(Dropout(0.01))
    model.add((Dense(1)))

    model.compile(loss='mean_squared_error', optimizer="adam")
    model.summary()
    callback = EarlyStopping(monitor='loss', patience=20, verbose=1, mode='min')
    model.fit(x_train, y_train, epochs=20, batch_size=5, callbacks=[callback])
    lstm_pre = model.predict(x_test)
    lstm_pre = pd.DataFrame(lstm_pre.flatten(), columns=["sbi"])
    data_frame = data_process.denormalize(lstm_pre, big, small)

    data_frame.to_csv(f'{station_id}_public.csv', index=False)
    print(data_frame)