import numpy as np
from liblinear.liblinearutil import *

path = "hw4_train.dat.txt"
data = []
yn = []
x = []


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
            numbers.insert(6, numbers[1] * numbers[i+1])
        for i in range(4):
            numbers.insert(6, numbers[2] * numbers[i+2])
        for i in range(3):
            numbers.insert(6, numbers[3] * numbers[i+3])
        for i in range(2):
            numbers.insert(6, numbers[4] * numbers[i+4])

        numbers.insert(6, numbers[5] * numbers[5])
        # numbers.insert(6, numbers[0] * numbers[1] )
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
        yn.append(numbers[len(numbers)-1])


except IOError:
    print('ERROR: can not found ' + path)
    if f:
        f.close()

finally:
    if f:
        f.close()

print(x[0])


prob = problem(yn, x)
Ein_a = []

# 定义 lambda 值的范围
lambda_values = [-6, -4, -2, 0, 2]

best_lambda = None
best_log10_lambda = float('inf')  # 初始化为正无穷大

for log10_lambda in lambda_values:
    C = 10**log10_lambda
    print(1/(2*C))
    param = parameter(f'-s 0 -c {1/(2*C)} -e 0.000001 -q')

    # 训练模型
    model = train(prob, param)
    # 计算训练错误 Ein
    best_Ein = float('inf')
    print("lamda" + str(log10_lambda))
    p_label, p_acc, p_vals = predict(yn, x, model)
    Ein = np.mean(np.array(p_label) != np.array(yn))

    Ein_a.append(Ein)

min_Ein = min(Ein_a)
print("Best Ein:", min_Ein)
print("log10(λ*):", lambda_values[Ein_a.index(min_Ein)])