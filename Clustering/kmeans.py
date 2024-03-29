from sklearn.datasets import load_iris
import random
import pandas as pd

def euclidean_squared(p1, p2):
    return sum(
        (val1 - val2) ** 2
        for val1, val2 in zip(p1, p2)
    )


class Kmeans:
    def __init__(self, k, distance, max_iters, use_range=True):
        self.k = k
        self.distance = distance
        self.use_range = use_range
        self.max_iters = max_iters
        self.centroids = []
        self.inertia_ = 0

    # Crear un centroide random
    def _get_range_random_value(self, points, feature_idx):
        feat_values = [point[feature_idx] for point in points]
        feat_max = max(feat_values)
        feat_min = min(feat_values)
        return random.random() * (feat_max - feat_min) + feat_min

    # Crear centroides random
    def _create_random_centroids(self, rows):
        n_feats = len(points[0])
        for cluster_idx in range(self.k):
            point = [0.0] * n_feats
            for feature_idx in range(n_feats):
                point[feature_idx] = self._get_range_random_value(points, feature_idx)
            self.centroids.append(point)

    def _create_points_centroids(self, points):
        raise NotImplementedError


    def _find_closest_centroid(self, row):
        min_dist = 2 ** 64
        closest_centroid_idx = None

        for centroid_idx, centroid in enumerate(self.centroids):
            dist = self.distance(row, centroid)
            if dist < min_dist:
                closest_centroid_idx = centroid_idx
                min_dist = dist
        return closest_centroid_idx, min_dist


    def _average_points(self, points_in_cl):
        avgs = []
        for i in range(len(points_in_cl[0])):
            avgs.append(sum([p[i] for p in points_in_cl]) / float(len(points)))
        return avgs


    def _update_centroids(self, matches, rows):
        for cluster_idx in range(len(matches)):
            points_2 = [rows[i] for i in matches[cluster_idx]]
            if not points_2:
                continue
            avrg = self._average_points(points_2)
            self.centroids[cluster_idx] = avrg

    def fit(self, rows):

        lastdistance=2 ** 64
        mejorDistancia = -10
        totalbestmatches = None
        for i in range(25):
            self.centroids = []
            if self.use_range:
                self._create_random_centroids(rows)
            else:
                self._create_points_centroids(rows)

            lastmatches = None
            distance = 0.0

            for iteration in range(self.max_iters):
                bestmatches = [[] for _ in range(self.k)]

                for row_idx, row in enumerate(rows):
                    centroid, dist = self._find_closest_centroid(row)
                    bestmatches[centroid].append(row_idx)

                if bestmatches == lastmatches:
                    break
                lastmatches = bestmatches


                self._update_centroids(bestmatches, rows)


            for row_idx, row in enumerate(rows):
                centroid, dist = self._find_closest_centroid(row)
                distance += dist

            if distance > lastdistance:
               mejorDistancia = lastdistance
               totalbestmatches = bestmatches
            lastdistance = distance


        if mejorDistancia == -10:
            mejorDistancia = distance
            totalbestmatches = bestmatches

        self.inertia = mejorDistancia

        return totalbestmatches, mejorDistancia


    def predict(self, rows):
        predictions = list(map(self._find_closest_centroid, rows))
        return predictions

def read_file(file_path, data_sep=",", ignore_first_line=False):
    prototypes = []
    # Open file
    with open(file_path, "r") as fh:
        # Strip lines
        strip_reader = (line.strip() for line in fh)

        # Filter empty lines
        filtered_reader = (line for line in strip_reader if line)

        # Skip first line if needed
        if ignore_first_line:
            next(filtered_reader)

        # Split line, parse token and append to prototypes
        for line in filtered_reader:
            prototypes.append([filter_token(token) for token in line.split(data_sep)])

    return prototypes



def filter_token(token):
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return token



if __name__ == "__main__":

    points = [
        [1, 1],
        [2, 1],
        [4, 3],
        [5, 4]
    ]

    points2 = [
        [2, 4],
        [3, 5],
        [3, 2],
        [5, 2],
        [5, 4],
        [7, 3],
        [7, 8],
        [8, 4],
    ]

    centroids = [
        [1, 1],
        [2, 1]
    ]


    kmeans = Kmeans(k=2, distance=euclidean_squared, max_iters=5)
    bestmatches, dist = kmeans.fit(points2)
    print(bestmatches)
    print(kmeans.predict(points2))
    print(dist)

    kmeans = Kmeans(k=2, distance=euclidean_squared, max_iters=5)
    bestmatches, dist = kmeans.fit(points)
    print(bestmatches)
    print(kmeans.predict(points))
    print(dist)

    print("\n Fichero Ceeds:")
    distancia = [[] for _ in range(8)]
    df = pd.DataFrame(columns=['Distancia Total ', 'Distancia por cada row'])
    for cont in range(8):
        kmeans = Kmeans(k=cont + 1, distance=euclidean_squared, max_iters=5)
        fitx = read_file(file_path="seeds.csv", data_sep=",")
        bestmatches, dist = kmeans.fit(rows=fitx)
        if cont < 3:
            print("\n Con K =", cont + 1, ":\n----------------- \n",
                  kmeans.predict(fitx), "\n --------------- ")

        df.loc[cont] = [dist, kmeans.predict(fitx)]
    print(df)
