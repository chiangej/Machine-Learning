from libsvm.svmutil import *
from matplotlib import pyplot as plt

path = "training_data.txt"
path2 = "testing_data.txt"

y, x = svm_read_problem(path)
test_y, test_x = svm_read_problem(path2)
y_binary = [1 if label == 1 else -1 for label in y]
y_binary_test = [1 if label == 1 else -1 for label in test_y]
C_values = [0.01, 0.1, 1, 10, 100]

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

    # Use svm_predict to get the predictions on a test set (replace test_y and test_x with your test data)
    p_label, p_acc, p_val = svm_predict(y_binary_test, test_x, m)

    # Extract Eout from p_acc (the accuracy)
    Eout = 100 - p_acc[0]

    # Update the minimum Eout and the corresponding C value
    if Eout < min_Eout:
        min_Eout = Eout
        best_C_value = C

# Output the minimum Eout and the corresponding C value
print(f"最小 Eout: {min_Eout}%")
print(f"对应的 C 值: {best_C_value}")

