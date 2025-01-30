import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def squared_euclidean(X, Y):
    return np.sum((X - Y) ** 2)

def kl_divergence(X, Y):
    return np.dot(X, np.log2(X/Y))

def itakura_saito(X, Y):
    return np.sum(X / Y - np.log(X / Y) - 1)

def generate_gaussian_clusters(n_clusters, points_per_cluster, centers, std=1):
    data = []
    for i in range(n_clusters):
        mean = centers[i]
        cov = np.eye(mean.shape[0]) * std
        data_cluster = np.random.multivariate_normal(mean, cov, points_per_cluster)
        data.append(data_cluster)
    return np.concatenate(data)

def plot_clusters(data, labels=None, centroids=None):
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(np.unique(labels)))]
    if (labels is None):
        plt.scatter(data[:, 0], data[:, 1])
    else:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            cluster_points = data[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[label], label=f'Cluster {label}')
    if (centroids is not None):
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="*", color="black", s=200, label='Centroids')

    plt.legend()
    plt.show()

def plot_clusters_3d(data, labels=None, centroids=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    if labels is None:
        ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    else:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            ax.scatter(data[labels == label, 0], data[labels == label, 1], data[labels == label, 2], label=f'Cluster {label}')
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='*', s=300, c='black', label='Centroids')

    ax.legend()
    plt.show()
