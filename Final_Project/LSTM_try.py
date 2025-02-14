import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.src.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

root_directory = r"C:\Users\chiangej\Desktop\data\html.2023.final.data\release"

date_folder = '20231002'
station_id = '500101001'
file_path = os.path.join(root_directory, date_folder, f"{station_id}.json")
print(file_path, )
if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        station_info = json.load(file)
        # print(station_info)

# 一開始的dictionary是 station_info
keys_to_delete = [key for key, value in station_info.items() if not value]

for key in keys_to_delete:
    del station_info[key]

x = []
x1 = []
for time_key in station_info.keys():
    hour, minute = map(int, time_key.split(':'))  # 將時間字串分割成小時和分鐘
    total_minutes = 60 * hour + minute  # 轉換成總共的分鐘數
    x1.append(total_minutes)
    x.append(total_minutes)
    x1 = []

y = []
for key, value in station_info.items():
    y.append(value['sbi'])

data = pd.DataFrame({'time': x, 'amount': y})

print(y)
print(x)
print(data)

# 正規化(normalize) 資料，使資料值介於[0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
x_scale = scaler.fit_transform(np.array(x).reshape(-1, 1))

print(x_scale)
print(np.array(x_scale))

# 将输入序列调整为适用于 LSTM 模型的 3D 形状。
# 其中，第一个维度是样本数量，第二个维度是时间步长（60 天），第三个维度是特征数量（1，因为只有一个特征，即开盘价）。
train_x = np.reshape(x_scale, (x_scale.shape[0], 1, x_scale.shape[1]))
print(x_scale.shape[0])
print(x_scale.shape[1])
look_back = 1

# 建立及訓練 LSTM 模型
model = Sequential()
model.add(LSTM(4, return_sequences = True,input_shape=(1, look_back)))
model.add(Dropout(0.2))


# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(4))
model.add(Dropout(0.2))

model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_x, np.array(y), epochs=100, batch_size=1, verbose=2)

# 預測
trainPredict = model.predict(train_x)

# 回復預測資料值為原始數據的規模
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([y])

# 畫訓練資料趨勢圖
print(trainPredict)

# Visualising the results
plt.plot(trainY.flatten(), color='red', label='Real Google Stock Price')  # 紅線表示真實股價
plt.plot(trainPredict, color='blue', label='Predicted Google Stock Price')  # 藍線表示預測股價
plt.title('Ubike Prediction')
plt.xlabel('Time')
plt.ylabel('num of bike')
plt.legend()
plt.show()
