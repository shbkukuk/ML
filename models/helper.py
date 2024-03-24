def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset)):
        col_values = dataset[i]
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
    fold_size = int(len(dataset)/)