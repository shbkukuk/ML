from math import sqrt


class NearestNeighbour:
    def __init__(self, k: int, distance_method: str, train, test):
        self.k = k
        if distance_method == 'euclidean':
            self.distance = self._euclidean()
        elif distance_method == 'manhattan':
            self.distance = self._manhattan()
        else:
            raise NotImplementedError
        self.train = train
        self.test = test

    def _euclidean(self, row1, row2):
        distance = 0.0
        for i in range(len(row1) - 1):
            distance += (row1[i] - row2[i]) ** 2

        return sqrt(distance)

    def _manhattan(self):
        return None

    def _get_neighbors(self):
        distances = list()
        for train_row in self.train:
            dist = self.distance(train_row, self.test)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(self.k):
            neighbors.append(distances[i][0])
        return neighbors

    def predict_classification(self):
        neighbors = self._get_neighbors()
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        return prediction
