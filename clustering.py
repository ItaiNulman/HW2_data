import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(2)


def add_noise(data):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :return: data + noise, where noise~N(0,0.001^2)
    """
    noise = np.random.normal(loc=0, scale=0.001, size=data.shape)
    return data + noise


def choose_initial_centroids(data, k):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :param k: number of clusters
    :return: numpy array of k random items from dataset
    """
    n = data.shape[0]
    indices = np.random.choice(range(n), k, replace=False)
    return data[indices]


# ====================
def transform_data(df, features):
    """
    Performs the following transformations on df:
        - selecting relevant features
        - scaling
        - adding noise
    :param df: dataframe as was read from the original csv.
    :param features: list of 2 features from the dataframe
    :return: transformed data as numpy array of shape (n, 2)
    """
    # numpy array of shape (n, 2)
    transformed_data = df[features].to_numpy()
    sum_feature_1 = transformed_data[:, 0].sum()
    sum_feature_2 = transformed_data[:, 1].sum()
    min_feature_1 = min(transformed_data[:, 0])
    min_feature_2 = min(transformed_data[:, 1])
    transformed_data[:, 0] = (transformed_data[:, 0] - min_feature_1) / sum_feature_1
    transformed_data[:, 1] = (transformed_data[:, 1] - min_feature_2) / sum_feature_2
    transformed_data = add_noise(transformed_data)
    return transformed_data


def kmeans(data, k):
    """
    Running kmeans clustering algorithm.
    :param data: numpy array of shape (n, 2)
    :param k: desired number of cluster
    :return:
    * labels - numpy array of size n, where each entry is the predicted label (cluster number)
    * centroids - numpy array of shape (k, 2), centroid for each cluster.
    """
    prev_centroids = choose_initial_centroids(data, k)
    while True:
        labels = assign_to_clusters(data, prev_centroids)
        current_centroids = recompute_centroids(data, labels, k)
        if np.array_equal(prev_centroids, current_centroids):
            break
        prev_centroids = current_centroids

    return labels, current_centroids


def visualize_results(data, labels, centroids, path):
    """
    Visualizing results of the kmeans model, and saving the figure.
    :param data: data as numpy array of shape (n, 2)
    :param labels: the final labels of kmeans, as numpy array of size n
    :param centroids: the final centroids of kmeans, as numpy array of shape (k, 2)
    :param path: path to save the figure to.
    """
    pass
    # plt.savefig(path)


def dist(x, y):
    """
    Euclidean distance between vectors x, y
    :param x: numpy array of size n
    :param y: numpy array of size n
    :return: the euclidean distance
    """
    res = 0
    for a, b in zip(x, y):
        res += (a - b) ** 2
    return res ** 0.5


def assign_to_clusters(data, centroids):
    """
    Assign each data point to a cluster based on current centroids
    :param data: data as numpy array of shape (n, 2)
    :param centroids: current centroids as numpy array of shape (k, 2)
    :return: numpy array of size n
    """
    n = data.shape[0]
    k = centroids.shape[0]
    labels = np.zeros(n)
    for i in range(n):
        min_dist = dist(data[i], centroids[0])
        min_index = 0
        for j in range(k):
            if dist(data[i], centroids[j]) < min_dist:
                min_dist = dist(data[i], centroids[j])
                min_index = j
        labels[i] = min_index

    return labels


def recompute_centroids(data, labels, k):
    """
    Recomputes new centroids based on the current assignment
    :param data: data as numpy array of shape (n, 2)
    :param labels: current assignments to clusters for each data point, as numpy array of size n
    :param k: number of clusters
    :return: numpy array of shape (k, 2)
    """
    new_centroids = np.zeros((k, 2))
    n = labels.shape[0]
    for i in range(k):
        count = 0
        for j in range(n):
            if labels[j] == i:
                new_centroids[i] += data[j]
                count += 1
        new_centroids[i] /= count

    return new_centroids
