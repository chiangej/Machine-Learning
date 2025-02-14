import pandas as pd
import data_process
import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.src.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense

station = []
with open(r"C:\Users\chiangej\PycharmProjects\ML_Final_Project\html.2023.final.data\sno_test_set.txt", 'r', encoding='utf-8') as file:
    lines = file.readlines()
    station.append(lines)
station = [item.strip() for sublist in station for item in sublist]

for j in range(0, 39):
    station_id = station[j]

    dfs = []
    result = pd.DataFrame()
    #取10月資料(從25號開始)
    for i in range(7):
        day = 25 + i
        one_day = data_process.read_data(f"202310{day}", station_id)
        dfs.append(one_day)

    for i in range(30):
        day = i + 1
        if i <= 8:
            one_day = data_process.read_data(f"2023110{day}", station_id)
        elif day == 30:
            one_day = data_process.read_data("20231127", station_id)
        else:
            one_day = data_process.read_data(f"202311{day}", station_id)
        dfs.append(one_day)

    for i in range(15):
        day = i + 1
        if i <= 8:
            one_day = data_process.read_data(f"2023120{day}", station_id)
        else:
            one_day = data_process.read_data(f"202312{day}", station_id)
        dfs.append(one_day)

    one_day = data_process.read_data("20231209", station_id)
    second_day = data_process.read_data("20231210", station_id)
    dfs.append(one_day)
    dfs.append(second_day)
    result = pd.concat(dfs, ignore_index=True)
    big, small = result["sbi"].max(), result["sbi"].min()
    data_nor = data_process.normalize(result)

    data_num = 721
    x_train = []
    y_train = []
    print(data_num)
    for i in range(data_num):
        subset_df = data_nor.head(1008)
        x_train.append(subset_df.head(504).values.tolist())
        y_train.append(subset_df.tail(504)["sbi"].values.tolist())
        data_nor = data_nor.iloc[4:]
    null_values = data_nor.isnull().sum()

    # 打印出有空值的列
    print("含有空值的列:")
    print(null_values[null_values > 0])

    x_test = data_nor.tail(504).values.tolist()
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (1, 504, 6))
    y_train = np.reshape(y_train, (721, 504))
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

    data_frame.to_csv(f'{station_id}_private.csv', index=False)
    print(data_frame)