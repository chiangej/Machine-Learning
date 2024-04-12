import random
import numpy as np
import matplotlib.pyplot as plt

from liblinear.liblinearutil import *
min_lamda_cv = []
min_lamda = []
for p in range(128):
    path = "hw4_train.dat.txt"
    data = []
    y_train_v6 = []
    data_v6 = []
    min_Ein = []
    Ecv = []
    # np.random.seed(p)
    aa = []

    try:
        f = open(path, 'r')
        for line in f.readlines():
            text = []
            numbers = [float(num) for num in line.strip().split()]

            for i in range(6):
                b = numbers[i] ** 2 * numbers[0]
                numbers.insert(6, b)
            for i in range(6):
                b = numbers[i] ** 2 * numbers[1]
                numbers.insert(6, b)
            for i in range(6):
                b = numbers[i] ** 2 * numbers[2]
                numbers.insert(6, b)
            for i in range(6):
                b = numbers[i] ** 2 * numbers[3]
                numbers.insert(6, b)
            for i in range(6):
                b = numbers[i] ** 2 * numbers[4]
                numbers.insert(6, b)
            for i in range(6):
                b = numbers[i] ** 2 * numbers[5]
                numbers.insert(6, b)

            for i in range(6):
                numbers.insert(6, numbers[0] * numbers[i])
            for i in range(5):
                numbers.insert(6, numbers[1] * numbers[i + 1])
            for i in range(4):
                numbers.insert(6, numbers[2] * numbers[i + 2])
            for i in range(3):
                numbers.insert(6, numbers[3] * numbers[i + 3])
            for i in range(2):
                numbers.insert(6, numbers[4] * numbers[i + 4])

            numbers.insert(6, numbers[5] * numbers[5])
            numbers.insert(6, numbers[0] * numbers[1] * numbers[2])
            numbers.insert(6, numbers[0] * numbers[1] * numbers[3])
            numbers.insert(6, numbers[0] * numbers[1] * numbers[4])
            numbers.insert(6, numbers[0] * numbers[1] * numbers[5])
            numbers.insert(6, numbers[0] * numbers[2] * numbers[3])
            numbers.insert(6, numbers[0] * numbers[2] * numbers[4])
            numbers.insert(6, numbers[0] * numbers[2] * numbers[5])
            numbers.insert(6, numbers[0] * numbers[3] * numbers[4])
            numbers.insert(6, numbers[0] * numbers[3] * numbers[5])
            numbers.insert(6, numbers[0] * numbers[4] * numbers[5])
            numbers.insert(6, numbers[1] * numbers[2] * numbers[3])
            numbers.insert(6, numbers[1] * numbers[2] * numbers[4])
            numbers.insert(6, numbers[1] * numbers[2] * numbers[5])
            numbers.insert(6, numbers[1] * numbers[3] * numbers[4])
            numbers.insert(6, numbers[1] * numbers[3] * numbers[5])
            numbers.insert(6, numbers[1] * numbers[4] * numbers[5])
            numbers.insert(6, numbers[2] * numbers[3] * numbers[4])
            numbers.insert(6, numbers[2] * numbers[3] * numbers[5])
            numbers.insert(6, numbers[2] * numbers[4] * numbers[5])
            numbers.insert(6, numbers[3] * numbers[4] * numbers[5])
            numbers.insert(0, 1)

            aa.append(numbers)

    except IOError:
        print('ERROR: can not found ' + path)

    finally:
        if f:
            f.close()

    random.shuffle(aa)
    for u in range(200):
        data_v6.append(aa[u][:-1])
        y_train_v6.append(aa[u][84])

    data_v1, y_train_v1 = data_v6[0:40:1], y_train_v6[0:40:1]
    data_v2, y_train_v2 = data_v6[40:80:1], y_train_v6[40:80:1]
    data_v3, y_train_v3 = data_v6[80:120:1], y_train_v6[80:120:1]
    data_v4, y_train_v4 = data_v6[120:160:1], y_train_v6[120:160:1]
    data_v5, y_train_v5 = data_v6[160:200:1], y_train_v6[160:200:1]

    # 定义 lambda 值的范围
    lambda_values = [-6, -4, -2, 0, 2]
    Ein_v1 = []

    best_lambda = None
    best_log10_lambda = float('inf')  # 初始化为正无穷大

    for log10_lambda in lambda_values:
        C = 10 ** log10_lambda
        param = parameter(f'-s 0 -c {1 / (2 * C)} -e 0.000001 -q')

        # 训练模型
        x = data_v6[40:200:1]
        y = y_train_v6[40:200:1]
        prob_v1 = problem(y,x)
        model = train(prob_v1, param)
        # 计算训练错误 Ein
        best_Ein = float('inf')
        data_v1, y_train_v1 = data_v6[0:40:1], y_train_v6[0:40:1]

        p_label, p_acc, p_vals = predict(y_train_v1, data_v1, model)
        Ein = np.mean(np.array(p_label) != np.array(y_train_v1))
        Ein_v1.append(Ein)

    Ein_v2 = []
    min_Ein_v2 = []
    min_lamda_v2 =[]
    for log10_lambda in lambda_values:
        x = []
        y = []
        C = 10 ** log10_lambda
        param = parameter(f'-s 0 -c {1 / (2 * C)} -e 0.000001 -q')


        # 计算训练错误 Ein
        best_Ein = float('inf')
        for i in range(40):
            y.append(y_train_v1[i])
            y.append(y_train_v3[i])
            y.append(y_train_v4[i])
            y.append(y_train_v5[i])
            x.append(data_v1[i])
            x.append(data_v3[i])
            x.append(data_v4[i])
            x.append(data_v5[i])

        # 训练模型
        prob_v2 = problem(y,x)
        model = train(prob_v2, param)
        data_v2, y_train_v2 = data_v6[40:80:1], y_train_v6[40:80:1]

        p_label, p_acc, p_vals = predict(y_train_v2, data_v2, model)
        Ein = np.mean(np.array(p_label) != np.array(y_train_v2))
        Ein_v2.append(Ein)


    Ein_v3 = []
    min_Ein_v3 = []
    min_lamda_v3 = []
    for log10_lambda in lambda_values:
        x = []
        y = []
        C = 10 ** log10_lambda
        param = parameter(f'-s 0 -c {1 / (2 * C)} -e 0.000001 -q')

        # 计算训练错误 Ein
        best_Ein = float('inf')
        for i in range(40):
            y.append(y_train_v1[i])
            y.append(y_train_v2[i])
            y.append(y_train_v4[i])
            y.append(y_train_v5[i])
            x.append(data_v1[i])
            x.append(data_v2[i])
            x.append(data_v4[i])
            x.append(data_v5[i])

        # 训练模型
        prob_v3 = problem(y,x)
        model = train(prob_v3, param)
        data_v3, y_train_v3 = data_v6[80:120:1], y_train_v6[80:120:1]

        p_label, p_acc, p_vals = predict(y_train_v3, data_v3, model)
        Ein = np.mean(np.array(p_label) != np.array(y_train_v3))
        Ein_v3.append(Ein)


    Ein_v4 = []
    min_Ein_v4 = []
    min_lamda_v4 = []
    for log10_lambda in lambda_values:
        x = []
        y = []
        C = 10 ** log10_lambda
        param = parameter(f'-s 0 -c {1 / (2 * C)} -e 0.000001 -q')


        # 计算训练错误 Ein
        best_Ein = float('inf')
        for i in range(40):
            y.append(y_train_v1[i])
            y.append(y_train_v3[i])
            y.append(y_train_v2[i])
            y.append(y_train_v5[i])
            x.append(data_v1[i])
            x.append(data_v3[i])
            x.append(data_v2[i])
            x.append(data_v5[i])

        # 训练模型
        prob_v4 = problem(y,x)
        model = train(prob_v4, param)
        data_v4, y_train_v4 = data_v6[120:160:1], y_train_v6[120:160:1]
        p_label, p_acc, p_vals = predict(y_train_v4, data_v4, model)
        Ein = np.mean(np.array(p_label) != np.array(y_train_v4))
        Ein_v4.append(Ein)


    Ein_v5 = []
    min_Ein_v5 = []
    min_lamda_v5 = []
    for log10_lambda in lambda_values:
        x = []
        y = []
        C = 10 ** log10_lambda
        param = parameter(f'-s 0 -c {1 / (2 * C)} -e 0.000001 -q')


        # 计算训练错误 Ein
        best_Ein = float('inf')
        for i in range(40):
            y.append(y_train_v1[i])
            y.append(y_train_v3[i])
            y.append(y_train_v2[i])
            y.append(y_train_v4[i])
            x.append(data_v1[i])
            x.append(data_v3[i])
            x.append(data_v2[i])
            x.append(data_v4[i])

        # 训练模型
        prob_v4 = problem(y,x)
        model = train(prob_v4, param)
        data_v5, y_train_v5 = data_v6[160:200:1], y_train_v6[160:200:1]
        p_label, p_acc, p_vals = predict(y_train_v5, data_v5, model)
        Ein = np.mean(np.array(p_label) != np.array(y_train_v5))
        Ein_v5.append(Ein)


    for r in range (5):
        Ecv.append(Ein_v1[r]+Ein_v2[r]+Ein_v3[r]+Ein_v4[r]+Ein_v5[r])

    def find_max_index_of_min(arr):
        if not arr:
            return None  # 处理空数组的情况
        min_value = min(arr)  # 找到最小值
        indices_of_min = [i for i, x in enumerate(arr) if x == min_value]  # 找到所有最小值的索引
        # 取出索引值最大的
        max_index_of_min = max(indices_of_min)

        return max_index_of_min


    min_Ein.append(min(Ecv))
    min_lamda_cv.append(lambda_values[find_max_index_of_min(Ecv)])

    print(Ecv)


plt.hist(min_lamda_cv, bins=10)
plt.xlabel("log10(λ*)")
plt.ylabel("Frequency")
plt.show()
