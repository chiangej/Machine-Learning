import numpy as np


def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            items = line.strip().split()
            label = int(items[0])
            features = {int(item.split(':')[0]): float(item.split(':')[1]) for item in items[1:]}
            data.append((label, features))
    return data


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


def build_tree(data):
    if len(set(label for label, _ in data)) == 1:
        return {'leaf': True, 'value': data[0][0]}

    feature, threshold = find_best_split(data)
    if feature is None:
        return {'leaf': True, 'value': np.mean([label for label, _ in data])}

    left_data = [(label, feat) for label, feat in data if feat.get(feature, 0) <= threshold]
    right_data = [(label, feat) for label, feat in data if feat.get(feature, 0) > threshold]

    return {'leaf': False, 'feature': feature, 'threshold': threshold,
            'left': build_tree(left_data), 'right': build_tree(right_data)}


def predict(tree, features):
    if tree['leaf']:
        return tree['value']
    else:
        print(features.get((tree['feature']), 0))
        if features.get(tree['feature'], 0) <= tree['threshold']:
            return predict(tree['left'], features)
        else:
            return predict(tree['right'], features)


train_data = load_data('hw6_train.dat.txt')
test_data = load_data('hw6_test.dat.txt')

tree = build_tree(train_data)

train_predictions = [predict(tree, feat) for _, feat in train_data]
test_predictions = [predict(tree, feat) for _, feat in test_data]

train_labels = [label for label, _ in train_data]
test_labels = [label for label, _ in test_data]

train_error = np.mean((np.array(train_predictions) - np.array(train_labels)) ** 2)
test_error = np.mean((np.array(test_predictions) - np.array(test_labels)) ** 2)
print(f'Ein(g): {train_error}')
print(f'Eout(g): {test_error}')