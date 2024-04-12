import math

import numpy as np
import matplotlib.pyplot as plt

# Generate N training examples
N_train = 256
N_test = 4096
mean_pos = [3, 2]
cov_pos = [[0.4, 0], [0, 0.4]]
mean_neg = [5, 0]
cov_neg = [[0.6, 0], [0, 0.6]]

num_experiments = 128

lin_errors = []
logistic_errors = []

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(x, y, learning_rate, iterations):
    # Initialize weights with zeros

    w = np.zeros(3)

    for i in range(iterations):
        gradient = 0
        for j in range(N_train):
            scores = x[j]@w.T
            # Calculate the predicted probabilities
            probabilities = sigmoid(-scores * y[j])
            aa = (-x[j] * y[j]) * probabilities
            gradient += aa
            # Calculate the error (cross-entropy loss) and gradient

            # Update the weights using gradient descent
        w -= learning_rate * gradient/len(y)

    return w


# def logistic_error(w,x,y):
#     sum = 0
#     for i in range(len(y)):
#         print(-y[i] * np.array(x[i])*np.array(w))
#         sum += math.log(1+math.exp(-y[i] * np.dot(x[i],np.array(w))))
#     return sum/len(y)


def linear_regression(x, y):
    XTX_inv = np.linalg.inv(x.T @ x)
    w = XTX_inv @ x.T @ y
    return w

def zero_one_error(prediction, true_labels):
    misclassified = np.where(prediction != true_labels)[0]
    error = len(misclassified) / len(true_labels)
    return error


def generate_training_data(i):
    x_train = []
    y_train = []
    np.random.seed(i)
    for i in range(N_train):
        y = np.random.choice([1, -1])
        if y == 1:
            x = np.hstack([1, np.random.multivariate_normal(mean_pos, cov_pos)])

        else:
            x = np.hstack([1, np.random.multivariate_normal(mean_neg, cov_neg)])
        x_train.append(x)
        y_train.append(y)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return x_train,y_train

def generate_testing_data(i):
    np.random.seed()

    X_test = []
    y_test = []
    np.random.seed(i)
    for _ in range(N_test):
        if np.random.choice([1, -1]) == 1:
            x = np.hstack([1, np.random.multivariate_normal(mean_pos, cov_pos)])
            y = 1
        else:
            x = np.hstack([1, np.random.multivariate_normal(mean_neg, cov_neg)])
            y = -1
        X_test.append(x)
        y_test.append(y)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_test,y_test


for i in range(num_experiments):
    w = []
    gd = generate_training_data(1000*i)

    X_train = gd[0]
    y_train = gd[1]
    gt = generate_testing_data(1000*i)
    x_test = gt[0]
    y_test = gt[1]

    # Algorithm B: Logistic Regression with fixed learning rate
    w_log = logistic_regression(X_train, y_train, learning_rate=0.1, iterations=500)
    # print(w_log)
    predictions_b = np.sign(x_test @ w_log)
    error_b = zero_one_error(predictions_b, y_test)
    # error_a = logistic_error(w_log, x_test, y_test)
    logistic_errors.append(error_b)

    # Algorithm A: Linear Regression
    w_lin = linear_regression(X_train,y_train)
    predictions_a = np.sign(x_test @ w_lin)
    error_a = zero_one_error(predictions_a,y_test)
    lin_errors.append(error_a)
    print(i)
    print(error_a, error_b)

# Calculate the median E0/1
median_log = np.median(logistic_errors)
print(f"Median E0/1(wlog) over test experiments: {median_log}")

# Calculate the median Esqr
median_lin = np.median(lin_errors)
print(f"Median E0/1(wlin) over test experiments: {median_lin}")

# # Plot histogram of Esqr(wlin)
# plt.hist(lin_errors, bins=50)
# plt.xlabel("                 median:"+str(median_lin)+"E0/1(wlin)")
# plt.ylabel("Frequency")
# plt.title("Distribution of E0/1(wlin) over test experiments")
# plt.show()
#
# # Plot histogram of E0/1(wlin)
# plt.hist(logistic_errors, bins=50)
# plt.xlabel("                 median:"+str(median_log)+"E0/1(wlog)")
# plt.ylabel("Frequency")
# plt.title("Distribution of E0/1(wlog) over test experiments")
# plt.show()
#
# plt.scatter(lin_errors, logistic_errors, label='Scatter Plot')

# Add labels and a title
plt.xlabel('Eout_lin')
plt.ylabel('Eout_log')
plt.title('Simple Scatter Plot')

plt.scatter(lin_errors, logistic_errors, marker='o', s=20, alpha=0.5)
plt.xlabel('Linear Regression Errors (E0/1(wlin))')
plt.ylabel('Logistic Regression Errors (E0/1(wlog))')
plt.title('Q11 Scatter Plot of Linear vs. Logistic Regression Errors'+"\n"+"-median.lin"+str(median_lin)+"-median.log"+str(median_log))


plt.show()

