from libsvm.svmutil import *
import numpy as np

path = "training_data.txt"

y, x = svm_read_problem(path)
print(x)
y_binary = [1 if label == 4 else -1 for label in y]
C_values = [0.1, 1, 10]
Q_values = [2, 3, 4]
min_support_vectors = float('inf')
best_CQ_combination = None


for C in C_values:
    for Q in Q_values:
        # 设置 SVM 参数
        svm_param_str = f"-t 1 -d {Q} -c {C} -g 1 -r 1 "

        # 创建一个包含二元标签的问题
        prob = svm_problem(y_binary, x)

        # 训练 SVM 模型
        m = svm_train(prob, svm_parameter(svm_param_str))

        # 获取支持向量的数量
        num_support_vectors = m.get_nr_sv()

        # 更新最小的支持向量数量和对应的 (C, Q) 组合
        if num_support_vectors < min_support_vectors:
            min_support_vectors = num_support_vectors
            best_CQ_combination = (C, Q)



# m = svm_train(y, x, '-s 0 -c 31 -t 1 -d 3 -g 1 -r 1 -c 4 -b 0')
print(f"最小支持向量数量: {min_support_vectors}")
print(f"对应的 (C, Q) 组合: {best_CQ_combination}")
