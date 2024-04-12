import random
import numpy as np
import matplotlib.pyplot as plt
import liblinear
from liblinear.liblinearutil import *

min_Ein = []
min_lamda = []
for p in range(128):
    path = "hw4_train.dat.txt"
    data = []
    yn = []
    x = []
    random.seed(p)

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

            x.append(numbers[:-1])
            yn.append(numbers[len(numbers) - 1])


    except IOError:
        print('ERROR: can not found ' + path)
        if f:
            f.close()

    finally:
        if f:
            f.close()

    data_train = []
    y_train = []
    random.seed(p)
    for j in range(120):
        a = random.randint(0, len(x)-1)
        data_train.append(x[a])
        y_train.append(yn[a])
        x.pop(a)
        yn.pop(a)

    prob = problem(y_train, data_train)

    # 定义 lambda 值的范围
    lambda_values = [-6, -4, -2, 0, 2]
    Ein_a = []

    best_lambda = None
    best_log10_lambda = float('inf')  # 初始化为正无穷大

    for log10_lambda in lambda_values:
        C = 10 ** log10_lambda
        param = parameter(f'-s 0 -c {1 / (2 * C)} -e 0.000001 -q')

        # 训练模型
        model = train(prob, param)
        # 计算训练错误 Ein
        best_Ein = float('inf')
        p_label, p_acc, p_vals = predict(yn, x, model)
        Ein = np.mean(np.array(p_label) != np.array(yn))
        Ein_a.append(Ein)


    def find_max_index_of_min(arr):
        if not arr:
            return None  # 处理空数组的情况

        min_value = min(arr)  # 找到最小值
        indices_of_min = [i for i, x in enumerate(arr) if x == min_value]  # 找到所有最小值的索引

        # 取出索引值最大的
        max_index_of_min = max(indices_of_min)

        return max_index_of_min

    min_Ein.append(min(Ein_a))
    min_lamda.append(lambda_values[find_max_index_of_min(Ein_a)])
    print("Best Ein:", min(Ein_a))
    print("log10(λ*):", lambda_values[Ein_a.index(min(Ein_a))])

plt.hist(min_lamda, bins=10)
plt.xlabel("log10(λ*)")
plt.ylabel("Frequency")
plt.show()
