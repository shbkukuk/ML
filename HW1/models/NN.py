from math import sqrt


class NearestNeighbour:
    def __init__(self, k: int, distance_method: str, train, test):
        self.k = k
        if distance_method == 'euclidean':
            self.distance = self._euclidean
        elif distance_method == 'manhattan':
            self.distance = self._manhattan
        else:
            raise NotImplementedError
        self.train = train
        self.test = test

    def _euclidean(self, row1, row2):
        distance = 0.0
        for i in range(len(row1) - 1):
            distance += (row1[i] - row2[i]) ** 2

        return sqrt(distance)

    def _manhattan(self,row1,row2):
        distance = 0.0
        for i in range(len(row1)-1):
            distance += abs(row1[i] - row2[i])
        return distance

    def _get_neighbors(self,test_row):
        distances = list()
        for train_row in self.train:
            dist = self.distance(train_row, test_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(self.k):
            neighbors.append(distances[i][0])
        return neighbors

    def predict_classification(self):
        prediction = list()
        for test_row in self.test:
            neighbors = self._get_neighbors(test_row)
            output_values = [row[-1] for row in neighbors]
            output = max(set(output_values), key=output_values.count)
            prediction.append(output)
        return prediction

    def predict_regregision(self):
        prediction = list()
        for test_row in self.test:
            neighbors = self._get_neighbors(test_row)
            output_values = [row[-1] for row in neighbors]
            assert self.k == len(output_values), f"Neighbors {len(output_values)} and K {self.k} value are not same"
            output = sum(output_values) / self.k
            prediction.append(output)
        return prediction
