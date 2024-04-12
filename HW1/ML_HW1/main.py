import matplotlib.pyplot as plt
import random
import numpy as np
import random

path = "Data.txt"
data = []
w = np.zeros(13)

try:
    f = open(path, 'r')
    i = 0
    for line in f.readlines():

        text = []
        numbers = [float(num) for num in line.strip().split()]
        text.extend(numbers)
        data.append(text)
        i += i

except IOError:
    print('ERROR: can not found ' + path)
    if f:
        f.close()

finally:
    if f:
        f.close()

for array in data:
    array.insert(0, 1.0/11.26)

def random_data(w, fdata):

    misclassified = True
    i = 0
    N = 256
    ww = 0
    counts = 0
    w = np.zeros(13)

    while misclassified:
        misclassified = False
        # 判断是否误分类
        data = [x for x in fdata if len(x) > 8]

        # w1 = w- first data
        if np.any(w == 0):
            rdata1 = random.choice(data)
            dot_data = np.array(rdata1[:-1])*11.26
            yn = float(rdata1[13])
            w = w + yn * np.array(dot_data)
            misclassified = True

        rdata = random.choice(data)
        yn = np.double(np.array(rdata)[13])
        dot_data = np.array(rdata[:-1])*11.26
        w_xn = np.dot(dot_data, w)

        while yn * w_xn <= 0:
           # print(w_xn,yn)  # 內積 vs yn
            w = np.add(w, yn * np.array(dot_data))
            i = 0  # 連續正確 5N 次
            counts += 1  # 總共抓資料的次數
            ww += 1  # 錯誤的次數
            w_xn = np.dot(dot_data, w)

        if yn * w_xn >= 0:
            i += 1
            if i <= N*5:
                counts += 1
                misclassified = True
            else:
                return ww

uu = []
for i in range(1000):
    random.seed(i)
    uu.append(random_data(w, data))
median = np.median(uu)
print(median)

# 创建直方图
plt.hist(uu, bins=50, edgecolor='k', alpha=0.65)
# 设置标签和标题
plt.xlabel('Updates')
plt.ylabel('times')
plt.title('HW1-9_PLA')
# 显示直方图
plt.show()




