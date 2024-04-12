import numpy as np
from libsvm.svmutil import *
from matplotlib import pyplot as plt

path = "training_data.txt"
path2 = "testing_data.txt"

y, x = svm_read_problem(path)
test_y, test_x = svm_read_problem(path2)
y_binary = [1 if label == 3 else -1 for label in y]
y_binary_test = [1 if label == 3 else -1 for label in test_y]
C_values = [0.01, 0.1, 1, 10, 100]

w_norm_values = []

# Initialize variables to store the minimum Eout and corresponding C value
min_Eout = float('inf')
best_C_value = None

# Iterate over all values of C
for C in C_values:
    # Set up the SVM parameters with fixed gamma and the current C
    svm_param_str = f"-t 2 -g 1 -c {C} -q"

    # Create a problem using your data
    prob = svm_problem(y_binary, x)

    # Train the SVM model
    m = svm_train(prob, svm_parameter(svm_param_str))

    # 获取支持向量的索引
    SV_indices = m.get_sv_indices()

    # 获取支持向量的系数
    SV_coef = m.get_sv_coef()
    w = np.zeros(36)

    # 计算权重向量 w

    for i in range(len(SV_indices)):
        idx = SV_indices[i] - 1  # Adjust index since LIBSVM uses 1-based indexing
        xx = x[idx]

        for j in range(len(x[0])):
            #提取每个元素中冒号后面的值
            jj = str(j)
            values = xx.get(j, 0)
            w[j] += SV_coef[i][0] * values

    # 计算 ||w||
    w_norm = np.linalg.norm(w)
    print(w_norm)
    w_norm_values.append(w_norm)

    # Use svm_predict to get the predictions on a test set (replace test_y and test_x with your test data)
    p_label, p_acc, p_val = svm_predict(y_binary_test, test_x, m)

    # Extract Eout from p_acc (the accuracy)
    Eout = 100 - p_acc[0]



# Output the minimum Eout and the corresponding C value
print(f"最小 Eout: {min_Eout}%")
print(f"对应的 C 值: {best_C_value}")
print(f"||w||:{w_norm_values}")

cc = [-2,-1,0,1,2]

# Plot the line chart
plt.plot(cc, w_norm_values, marker='o')
plt.xlabel('C')
plt.ylabel('||w||')
plt.title('C versus ||w||')
plt.grid(True)
plt.show()