# 上禮拜對下禮拜一
from tensorflow.python.keras.metrics import MeanSquaredError

import data_process
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.src.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense

station_id = "500101002"

week1_mon = data_process.read_data("20231003")
week2_mon = data_process.read_data("20231010")
week3_mon = data_process.read_data("20231017")
# week4_mon = data_process.read_data("20231023", station_id)
week5_mon = data_process.read_data("20231031")
week6_mon = data_process.read_data("20231107")
week7_mon = data_process.read_data("20231114")
week8_mon = data_process.read_data("20231121")
week9_mon = data_process.read_data("20231128")

list1 = [week1_mon, week2_mon, week3_mon, week5_mon, week6_mon, week7_mon
    , week8_mon, week9_mon]

y_test = np.array(data_process.build_trainy(week9_mon))

y_train = []
for j in range(1, 6):
    y_train.append(data_process.build_trainy(list1[j]))

week1_mon_nor = data_process.normalize(week1_mon)
week2_mon_nor = data_process.normalize(week2_mon)
week3_mon_nor = data_process.normalize(week3_mon)
# week4_mon_nor = data_process.normalize(week4_mon)
week5_mon_nor = data_process.normalize(week5_mon)
week6_mon_nor = data_process.normalize(week6_mon)
week7_mon_nor = data_process.normalize(week7_mon)
week8_mon_nor = data_process.normalize(week8_mon)
week9_mon_nor = data_process.normalize(week9_mon)

nor_list = [week1_mon_nor, week2_mon_nor, week3_mon_nor, week5_mon_nor, week6_mon_nor, week7_mon_nor
    , week8_mon_nor, week9_mon_nor]

x_train = []
for i in range(5):
    x_train.append(data_process.build_trainx(nor_list[i]))

y_train_nor = []
scaler = MinMaxScaler(feature_range=(0, 1))
for i in range(1, 6):
    y_train = data_process.build_trainy(list1[i])
    y_train_nor.append(scaler.fit_transform(np.array(y_train).reshape(-1, 1)))

y_train_nor = np.array(y_train_nor)

x_train = np.array(x_train)
y_train = np.array(y_train)
y_test = np.array(y_test)
error = []


ll = [0.01, 0.02]

for first in ll:
    for second in ll:
        y_test_nor = scaler.fit_transform(np.array(y_test).reshape(-1, 1))
        x_test = data_process.build_trainx(nor_list[6])
        x_test = np.array(x_test)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 5))
        x_test = np.reshape(x_test, (1, x_test.shape[0], 5))
        model = Sequential()
        model.add(LSTM(4, input_shape=(x_train.shape[1], 5), return_sequences=True))
        model.add(Dropout(first))

        model.add(LSTM(4, return_sequences=True))
        model.add(Dropout(second))

        model.add((Dense(1)))  # or use model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer="adam")
        model.summary()
        history = model.fit(x_train, y_train_nor, epochs=300, batch_size=12)

        trainPredict = model.predict(x_test)
        # 回復預測資料值為原始數據的規模

        trainPredict_2d = trainPredict.reshape(-1, trainPredict.shape[-1])
        trainPredict_2d = scaler.inverse_transform(trainPredict_2d)
        y_test_nor = scaler.inverse_transform(y_test_nor)

        mse = MeanSquaredError()
        mse.update_state(y_test, trainPredict)
        print('Model Error (MSE):', mse.result().numpy())
        error.append(mse.result().numpy())


        plt.plot(trainPredict_2d, color='blue', label='Predicted')
        plt.plot(y_test_nor.flatten(), color='red', label='Real')  # 紅線表示真實股價# 藍線表示預測股價
        plt.title('Ubike Prediction on Tuesday <2, batch12, epoch300, Drop>' + str(first) + "/" + str(second))
        plt.xlabel('                     Time                           ' + station_id)
        plt.ylabel('num of bike')
        plt.legend()
        plt.show()

print(error)