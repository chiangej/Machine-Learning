import numpy as np

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            items = line.strip().split()
            label = float(items[0])
            features = {int(item.split(':')[0]): float(item.split(':')[1]) for item in items[1:]}
            data.append((label, features))
    return data


def squared_error(part, mean):
    labels = [label for label, _ in part]
    x = 0
    for labe in labels:
        x += (labe-mean) ** 2
    return x

def split(data_list, feature, threshold):
    less_than_threshold = [item for item in data_list if item[1].get(feature, 0) <= threshold]
    greater_than_threshold = [item for item in data_list if item[1].get(feature, 0) > threshold]
    return less_than_threshold, greater_than_threshold

def square_loss(data, feature, threshold):
    left = [(label, feat) for label, feat in data if feat.get(feature, 0) <= threshold]
    right = [(label, feat) for label, feat in data if feat.get(feature, 0) > threshold]

    if not left or not right:
        return float('inf')

    mean_left = np.mean([label for label, _ in left])
    mean_right = np.mean([label for label, _ in right])

    loss_left = np.sum([(label - mean_left) ** 2 for label, _ in left])
    loss_right = np.sum([(label - mean_right) ** 2 for label, _ in right])

    return loss_left + loss_right
def find_best_split(data):
    num_features = len(data[0][1])
    best_feature, best_threshold = None, None
    best_loss = float('inf')

    for feature in range(1, num_features + 1):
        feature_values = sorted(set(feat.get(feature, 0) for _, feat in data))
        for i in range(len(feature_values) - 1):
            threshold = 0.5 * (feature_values[i] + feature_values[i + 1])
            loss = square_loss(data, feature, threshold)

            if loss < best_loss:
                best_loss = loss

                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold

def build_tree(data_list):
    # 检查是否所有的标签都相同
    if len(set(x[0] for x in data_list)) == 1:
        return {'leaf': True, 'value': data_list[0][0]}

    best_feature, best_threshold = find_best_split(data_list)
    # 如果没有找到合适的特征进行分割
    if best_feature is None:
        print(x[0]for x in data_list)
        return {'leaf': True, 'value': np.mean([x[0] for x in data_list])}

    left, right = split(data_list, best_feature, best_threshold)

    # 递归构建左右子树
    return {
        'leaf': False,
        'feature': best_feature,
        'threshold': best_threshold,
        'left': build_tree(left),
        'right': build_tree(right)
    }

def predict(tree, features):
    if tree['leaf']:
        return tree['value']
    else:
        feature_value = (features.get(tree['feature'], 0))
        print(feature_value)
        if feature_value <= tree['threshold']:
            return predict(tree['left'], features)
        else:
            return predict(tree['right'], features)

def calculate_error(data_list, tree):
    predictions = [predict(tree, features) for _, features in data_list]
    labels = [label for label, _ in data_list]
    return np.mean((np.array(predictions) - np.array(labels)) ** 2)

# Load data
train_data = load_data('hw6_train.dat.txt')
test_data = load_data('hw6_test.dat.txt')


# Build tree
tree = build_tree(train_data)


# Calculate error
train_error = calculate_error(train_data,tree)
test_error = calculate_error(test_data, tree)
print(f'Ein(g): {train_error}')
print(f'Eout(g): {test_error}')
