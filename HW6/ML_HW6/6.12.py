import numpy as np
import requests
import matplotlib.pyplot as plt


def parse_libsvm_line(line):
    elements = line.split()
    target = float(elements[0])
    features = np.zeros(8)  # 根据数据集特征数量调整
    for e in elements[1:]:
        index, value = e.split(":")
        features[int(index) - 1] = float(value)
    return features, target


def download_data(url):
    response = requests.get(url)
    data = response.content.decode('utf-8').splitlines()
    features, targets = zip(*[parse_libsvm_line(line) for line in data])
    return np.array(features), np.array(targets)


def split_dataset(X, y, feature, threshold):
    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold
    return X[left_idx], y[left_idx], X[right_idx], y[right_idx]


def squared_error(y):
    return np.sum((y - np.mean(y)) ** 2)


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def get_potential_splits(X):
    potential_splits = {}
    n_features = X.shape[1]
    for feature_index in range(n_features):
        unique_values = np.unique(X[:, feature_index])
        potential_splits[feature_index] = (unique_values[:-1] + unique_values[1:]) / 2

    return potential_splits



def build_tree(X, y):
    num_samples, num_features = X.shape

    # 如果所有的樣本都是同一類別，或只有一個樣本，則返回葉節點
    if num_samples == 1 or len(np.unique(y)) == 1:
        leaf_value = np.mean(y)
        return TreeNode(value=leaf_value)

    best_feature, best_threshold, best_error = None, None, float('inf')
    potential_splits = get_potential_splits(X)

    for feature_index in potential_splits:
        for threshold in potential_splits[feature_index]:
            _, y_left, _, y_right = split_dataset(X, y, feature_index, threshold)
            error = squared_error(y_left) + squared_error(y_right)

            if error < best_error:
                best_error, best_feature, best_threshold = error, feature_index, threshold

    if best_feature is not None:
        X_left, y_left, X_right, y_right = split_dataset(X, y, best_feature, best_threshold)
        left_subtree = build_tree(X_left, y_left)
        right_subtree = build_tree(X_right, y_right)
        return TreeNode(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    # 如果無法進一步分割，返回葉節點
    return TreeNode(value=np.mean(y))


def predict(tree, X):
    if tree.value is not None:
        return tree.value
    feature_val = X[tree.feature]
    branch = tree.left if feature_val <= tree.threshold else tree.right
    return predict(branch, X)


def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=int(0.5 * n_samples), replace=True)
    return X[indices], y[indices]


def random_forest(X_train, y_train, X_test, y_test, n_trees):
    trees = []
    e_ins = []
    e_outs = []
    e_outs_forest = []

    for i in range(n_trees):
        X_sample, y_sample = bootstrap_sample(X_train, y_train)
        tree = build_tree(X_sample, y_sample)
        trees.append(tree)

        predictions_train = [predict(tree, x) for x in X_train]
        e_in = np.mean((y_train - predictions_train) ** 2)
        e_ins.append(e_in)

        predictions_test = [predict(tree, x) for x in X_test]
        e_out = np.mean((y_test - predictions_test) ** 2)
        e_outs.append(e_out)

        forest_predictions = [np.mean([predict(t, x) for t in trees]) for x in X_test]
        e_out_forest = np.mean((y_test - forest_predictions) ** 2)
        e_outs_forest.append(e_out_forest)
    return trees, e_ins, e_outs, e_outs_forest


def main(train_url, test_url):
    X_train, y_train = download_data(train_url)
    X_test, y_test = download_data(test_url)

    trees, e_ins, e_outs, e_outs_forest = random_forest(X_train, y_train, X_test, y_test, n_trees=2000)

    # 計算隨機森林的 E_in 和 E_out
    predictions_train = [np.mean([predict(tree, x) for tree in trees]) for x in X_train]
    e_in_G = np.mean((y_train - predictions_train) ** 2)

    predictions_test = [np.mean([predict(tree, x) for tree in trees]) for x in X_test]
    e_out_G = np.mean((y_test - predictions_test) ** 2)

    # 繪製 Eout 的直方圖
    plt.hist(e_outs, bins=20, edgecolor='black')
    plt.xlabel('Eout')
    plt.ylabel('Frequency')
    plt.title('HW6_10')

    # 繪製散點圖
    plt.figure()
    plt.scatter(e_ins, e_outs, alpha=0.5)
    plt.scatter([e_in_G], [e_out_G], color='red')  # 標記隨機森林的點
    plt.xlabel('Ein')
    plt.ylabel('Eout')
    plt.title('HW6_11')

    # 繪製 Eout(gt) 和 Eout(Gt) 作為 t 的函數
    plt.figure()
    plt.plot(e_outs, label='Eout_g')
    plt.plot(e_outs_forest, label='Eout_G')
    plt.xlabel('Number of Trees (t)')
    plt.ylabel('Eout')
    plt.title('HW6_12 Eout(gt) and Eout(Gt)')
    plt.legend()
    plt.show()


# 執行
train_url = 'http://www.csie.ntu.edu.tw/~htlin/course/ml23fall/hw6/hw6_train.dat'
test_url = 'http://www.csie.ntu.edu.tw/~htlin/course/ml23fall/hw6/hw6_test.dat'
main(train_url, test_url)