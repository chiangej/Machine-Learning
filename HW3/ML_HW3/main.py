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

# Perform 128 experiments
num_experiments = 128
esqr_values = []

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

    # Calculate squared in-sample error
    esqr = np.mean((X_train @ w_lin - y_train) ** 2)
    esqr_values.append(esqr)
    print(esqr)

# Plot histogram of Esqr(wlin)
plt.hist(esqr_values, bins=50)
plt.xlabel("Esqr(wlin)")
plt.ylabel("Frequency")
plt.title("Distribution of Esqr(wlin) over 128 experiments")
plt.show()

# Calculate the median Esqr
median_esqr = np.median(esqr_values)
print(f"Median Esqr(wlin) over 128 experiments: {median_esqr}")

