# 上禮拜對下禮拜一
import pandas as pd
from keras.src.callbacks import ModelCheckpoint

import data_process
import numpy as np
from keras.callbacks import  EarlyStopping
from keras.models import Sequential
from keras.src.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense

station = []
data_frame_final = pd.DataFrame(columns=["id", "sbi"])
with open(r"C:\Users\chiangej\Desktop\data\html.2023.final.data\sno_test_set.txt", 'r', encoding='utf-8') as file:
    lines = file.readlines()
    station.append(lines)
station = [item.strip() for sublist in station for item in sublist]

for station_id in station:
    print(station_id)
    week1_mon = data_process.read_data("20231002", station_id)
    week2_mon = data_process.read_data("20231009", station_id)
    week3_mon = data_process.read_data("20231016", station_id)
    # week5_mon = data_process.read_data("20231105", station_id)
    # week6_mon = data_process.read_data("20231112", station_id)
    # week7_mon = data_process.read_data("20231119", station_id)
    # week8_mon = data_process.read_data("20231126", station_id)
    # week9_mon = data_process.read_data("20231203", station_id)

    result = pd.concat([week1_mon, week3_mon])
    big, small = result["sbi"].max(), result["sbi"].min()
    data_nor = data_process.normalize(result)

    grouped = [data_nor.iloc[ i:i + 72, :] for i in range(0, 72*8, 72)] #要改
    nor_list = []
    for i, part in enumerate(grouped):
        nor_list.append(part.set_axis(['tot', 'sbi', 'bemp', "act", "min"], axis=1))

    x_train = []
    y_train = []
    x_test = [data_process.build_trainx(nor_list[7])]
    for i in range(len(nor_list) - 2): #要改
        x_train.append(data_process.build_trainx(nor_list[i]))
    for i in range(1, len(nor_list)-1): #要改
        y_train.append(data_process.build_trainy(nor_list[i]))

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 5))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 5))

    model = Sequential()
    model.add(LSTM(256, input_shape=(x_train.shape[1], 5), return_sequences=True))
    model.add(Dropout(0.05))
    model.add(LSTM(256, input_shape=(x_train.shape[1], 5), return_sequences=True))
    model.add(Dropout(0.01))
    model.add((Dense(1)))

    model.compile(loss='mean_squared_error', optimizer="adam")
    model.summary()
    # checkpoint = ModelCheckpoint('model.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
    callback = EarlyStopping(monitor='loss', patience=20, verbose=1, mode='min')
    model.fit(x_train, y_train, epochs=500, batch_size=5, callbacks=[callback])
    lstm_pre = model.predict(x_test)
    lstm_pre = pd.DataFrame(lstm_pre.flatten(), columns=["sbi"])
    data_frame = data_process.denormalize(lstm_pre, big, small)
    data = {'id': data_process.id_list(22, station_id), 'sbi': data_frame['sbi']} #要改
    df = pd.DataFrame(data)
    data_frame_final = pd.concat([data_frame_final, df])

data_frame_final.to_csv('20231022_layer2_dropout0.05/0.01.csv', index=False)
print(data_frame_final)

