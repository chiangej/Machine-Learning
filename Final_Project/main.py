import json
import os
import numpy as np

from libsvm.svm import svm_problem, svm_parameter
from libsvm.svmutil import *


root_directory = r"C:\Users\chiangej\Desktop\data\html.2023.final.data\release"

date_folder = '20231002'
station_id = '500101001'
file_path = os.path.join(root_directory, date_folder, f"{station_id}.json")
print(file_path, )
if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        station_info = json.load(file)
        # print(station_info)

#一開始的dictionary是 station_info
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

print(y)
print(x)
y_test = y[0:20:1]
x_test = x[0:20:1]

C_values = [0.1]
Q_values = [2]
min_support_vectors = float('inf')
best_CQ_combination = None

list_of_dicts = [{1: value} for value in x]
ls = [{1: value} for value in x_test]
print(list_of_dicts)


svm_param_str = f"-t 1 -s 3 -d 2 -c 0.001 -r 1 -h 0 -p 1 "

prob = svm_problem(y, list_of_dicts)


# 訓練 SVM 模型
m = svm_train(prob, svm_parameter(svm_param_str))
svm_predict(y_test, ls, m)

#  獲得支持向量的數量
num_support_vectors = m.get_nr_sv()

# 更新最小的支持向量數量和對應的 (C, Q) 组合
if num_support_vectors < min_support_vectors:
    min_support_vectors = num_support_vectors
    # best_CQ_combination = (C, Q)

print(f"最小支持向量數量: {min_support_vectors}")
print(f"對應的 (C, Q) 组合: {best_CQ_combination}")
