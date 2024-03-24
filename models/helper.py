from random import  randrange
from sklearn.tree import _tree
import  numpy as np
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset)):
        col_values = dataset[i,:]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return dataset

def str_to_int(dataset, column):
    for row in dataset:
        try:
            row[column] = int(row[column])
        except Exception:
            row[column] = 0
    return  dataset

def cross_validation(dataset,kfold):
    split_dataset = list()
    dataset_copy = dataset.copy()
    fold_size = int(len(dataset)/kfold)
    for _ in range(kfold):
        fold = list()
        while len(fold)<fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        split_dataset.append(fold)

    return  split_dataset

def confusion_matrix(actaul,predicted):
    num_classes = len(set(actaul))
    cm = [[0] * num_classes for _ in range(num_classes)]

    for true_label, pred_label in zip(actaul, predicted):
        cm[int(true_label)][int(pred_label)] += 1
    print_confusion_matrix(cm)
    print_metrics(actaul,predicted)
    return cm


def print_confusion_matrix(conf_matrix):
    num_classes = len(conf_matrix)
    print("Confusion Matrix:")
    print("True\Predicted\t", end="")
    for i in range(num_classes):
        print(i, end="\t")
    print()

    for i in range(num_classes):
        print(i, end="\t\t")
        for j in range(num_classes):
            print(conf_matrix[i][j], end="\t")
        print()



def print_metrics(y_true,y_pred):
    tp = tn = fp = fn = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1

    # Calculate precision, recall, and accuracy
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Print precision, recall, and accuracy
    print("Precision:", precision)
    print("Recall:", recall)
    print("Accuracy:", accuracy)
def tree_to_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node, rule):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            rule.append(f"{name} <= {threshold}")
            recurse(tree_.children_left[node], rule)
            rule.pop()
            rule.append(f"{name} > {threshold}")
            recurse(tree_.children_right[node], rule)
            rule.pop()
        else:
            rule.append(f"class: {np.argmax(tree_.value[node])}")
            rules.append(" and ".join(rule))
            rule.pop()

    rules = []
    recurse(0, [])
    return rules