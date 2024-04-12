import numpy as np
import matplotlib.pyplot as plt

# Generate N training examples
N = 256
mean_pos = [3, 2]
cov_pos = [[0.4, 0], [0, 0.4]]
mean_neg = [5, 0]
cov_neg = [[0.6, 0], [0, 0.6]]


# Implement linear regression
def linear_regression(x, y):
    XTX_inv = np.linalg.inv(x.T @ x)
    w = XTX_inv @ x.T @ y
    return w


def zero_one_error(prediction, true_labels):
    misclassified = np.where(prediction != true_labels)[0]
    error = len(misclassified) / len(true_labels)
    return error


# Perform 128 experiments
num_experiments = 128
w_lin_list = []
zero_one_errors = []

for i in range(num_experiments):
    np.random.seed(i)
    w_lin = []
    w = []
    X_train = []
    y_train = []
    for _ in range(N):
        y = np.random.choice([1, -1])
        if y == 1:
            x = np.hstack([1, np.random.multivariate_normal(mean_pos, cov_pos)])

        else:
            x = np.hstack([1, np.random.multivariate_normal(mean_neg, cov_neg)])
        X_train.append(x)
        y_train.append(y)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Train linear regression model
    w_lin = linear_regression(X_train, y_train)

    # Calculate 0/1 error for each experiment
    # Make predictions using w_lin
    predictions = np.sign(X_train @ w_lin)

    # Calculate 0/1 error
    error = zero_one_error(predictions, y_train)
    print(error)
    zero_one_errors.append(error)

# Plot histogram of E0/1(wlin)
plt.hist(zero_one_errors, bins=20)
plt.xlabel("E0/1(wlin)")
plt.ylabel("Frequency")
plt.title("Q10 Distribution of E0/1(wlin) over 128 experiments")
plt.show()

# Calculate the median E0/1
median_zo = np.median(zero_one_errors)
print(f"Median E0/1(wlin) over 128 experiments: {median_zo}")
