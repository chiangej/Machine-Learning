import random
import matplotlib.pyplot as plt
import numpy as np

from libsvm.svmutil import *

path = "training_data.txt"
path2 = "testing_data.txt"

y, x = svm_read_problem(path)
test_y, test_x = svm_read_problem(path2)
print(len(y))
y_binary = [1 if label == 1 else -1 for label in y]
y_binary_test = [1 if label == 1 else -1 for label in test_y]

C_values = [0.01, 0.1, 1, 10, 100]

# Initialize variables to store the minimum Eout and corresponding C value

c  = []

for i in range(1000):

    min_Eout = float('inf')
    best_C_value = None

    # random.seed(i)
    y_binary = np.array(y_binary)
    x = np.array(x)
    state = np.random.get_state()
    np.random.shuffle(y_binary)
    np.random.set_state(state)
    np.random.shuffle(x)
    y_binary = list(y_binary)
    x = list(x)
    y_binary1 = y_binary[0:200:1]
    y_binary2 = y_binary[200:4435:1]
    x1 = x[0:200:1]
    x2 = x[200:4435:1]

    for C in C_values:
        # Set up the SVM parameters with fixed gamma and the current C
        svm_param_str = f"-t 2 -g 1 -c {C} -q"

        # Create a problem using your data
        prob = svm_problem(y_binary2, x2)

        # Train the SVM model
        m = svm_train(prob, svm_parameter(svm_param_str))

        # Use svm_predict to get the predictions on a test set (replace test_y and test_x with your test data)
        p_label, p_acc, p_val = svm_predict(y_binary1, x1, m)

        # Extract Eout from p_acc (the accuracy)
        Eout = 100 - p_acc[0]

        # Update the minimum Eout and the corresponding C value
        if Eout < min_Eout:
            min_Eout = Eout
            best_C_value = C

    c.append(np.log10(best_C_value))
    print(f"最小 Eout: {min_Eout}%")
    print(f"对应的 C 值: {best_C_value}")

for i in range(5):
    c[i] = 10^c[i]


# 使用 plt.bar 画柱状图
plt.bar(range(len(c)), c, align='center', alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('log10(C)')
plt.show()
