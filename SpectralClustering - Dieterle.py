# Implementing Spectral Clustering for the class "PDEs on Graphs" in WS24/25
# Initial code taken from Professor Bungert
# Author: Priska Dieterle, Date: 11.11.2024

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import neighbors
from sklearn import cluster
from scipy import sparse
import pandas as pd
from pandas.plotting import scatter_matrix
import graphlearning as gl
import igraph as ig


def epsilon_ball_sparse(data, eps, n, sigma=1.):
    matrix_entries = []
    row = []
    col = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(data[i, :] - data[j, :], 2) < eps:
                matrix_entries += [similarity(data[i, :]/eps, data[j, :]/eps, sigma)]
                row += [i]
                col += [j]
    matrix_entries += matrix_entries
    row_once = row.copy()
    row += col
    col += row_once

    return sparse.coo_matrix((matrix_entries, (row, col)), shape=(n, n))


def similarity(x, y, sigma=1.):
    return np.exp(-np.linalg.norm(x-y)**2/(2*sigma**2))


def knn_sparse(data, k, n, sigma=1.):
    tree = neighbors.KDTree(data)
    dist, ind = tree.query(data, k=k + 1)  # +1 because it also sees itself at being the most similar
    matrix_entries = []
    row = []
    col = []
    for i in range(n):
        for j in range(i + 1, n):
            if j in ind[i] or i in ind[j]:
                matrix_entries += [similarity(data[i, :], data[j, :], sigma)]
                row += [i]
                col += [j]
    matrix_entries += matrix_entries
    row_once = row.copy()
    row += col
    col += row_once

    return sparse.coo_matrix((matrix_entries, (row, col)), shape=(n, n))


# Create plots for two clusters
def plot_points_with_labels(data, color):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=color)
    plt.axis('equal')
    plt.colorbar()


def plot_graph_with_labels(weight_matrix, plot_title, labels, coordinates):
    # Create the graph
    g = ig.Graph.Weighted_Adjacency(weight_matrix.toarray(), mode="undirected")

    # Create the plot
    color_dict = {0: "blue", 1: "pink"}

    # Plot the vertices at the point where they are, depending on x and y - coordinates
    layout = ig.Layout(coords=coordinates)

    fig, ax = plt.subplots()
    ig.plot(g,
            target=ax,
            vertex_color=[color_dict[label] for label in labels],  # give nodes colors acc. to labels
            vertex_size=10,
            edge_width=[edge for edge in g.es['weight']],
            layout=layout)
    plt.title(plot_title)


# Spectral Clustering for blobs
def spectral_clustering_for_blobs(epsilon=None, k=None, n = None, centers = None):
    if n is None:
        n = 1000
    if centers is None:
        centers = [(1, 1), (5, 1)]
    x, labels_blobs = datasets.make_blobs(n_samples=n, centers=centers)
    if epsilon is not None:
        weight_matrix = epsilon_ball_sparse(x, epsilon, n=n)
    elif k is not None:
        weight_matrix = knn_sparse(x, k, n, sigma=1.)
    else:
        raise Exception('Either epsilon or k has to be given')

    # Calculate the graph laplacian
    diagonal = np.squeeze(np.asarray(weight_matrix.sum(axis=0)))
    degree_matrix = sparse.coo_matrix((diagonal, (range(n), range(n))), shape=(n, n))
    laplacian = weight_matrix - degree_matrix

    vals, vecs = sparse.linalg.eigsh(-laplacian, M=degree_matrix, k=2, which="SM")

    # Plot the points/graph with the labels
    plot_points_with_labels(data=x, color=vecs[:, 1])
    labels = [1 if sign > 0 else 0 for sign in vecs[:, 1]]
    plot_graph_with_labels(weight_matrix, plot_title="Blobs", labels=labels, coordinates=x)


# Spectral Clustering for moons
def spectral_clustering_for_moons(epsilon=None, k=None, n=None):
    if n is None:
        n = 1000
    x, labels_moons = datasets.make_moons(n_samples=n, noise=0.08)

    # Create the weight matrix
    if epsilon is not None:
        weight_matrix = epsilon_ball_sparse(x, epsilon, n=n)
    elif k is not None:
        weight_matrix = knn_sparse(x, k, n, sigma=1.)
    else:
        raise Exception('Either epsilon or k has to be given')

    # Calculate the graph laplacian
    diagonal = np.squeeze(np.asarray(weight_matrix.sum(axis=0)))
    degree_matrix = sparse.coo_matrix((diagonal, (range(n), range(n))), shape=(n, n))
    laplacian = weight_matrix - degree_matrix

    vals, vecs = sparse.linalg.eigsh(-laplacian, M=degree_matrix, k=2, which="SM")

    # Plot the points/graph with the labels
    plot_points_with_labels(data=x, color=vecs[:, 1])
    labels = [1 if sign > 0 else 0 for sign in vecs[:, 1]]
    plot_graph_with_labels(weight_matrix, plot_title="Moons", labels=labels, coordinates=x)


# Spectral Clustering for digits
def spectral_clustering_for_digits(epsilon=None, k=None, only_some_digits=None):
    digits, digits_target = datasets.load_digits(return_X_y=True, as_frame=True)
    if only_some_digits is None:
        cluster_number = 10
        labels_original = digits_target.to_numpy()
    else:
        filtering_condition = digits_target.isin(only_some_digits)
        digits = digits[filtering_condition]
        labels_original = digits_target[filtering_condition].to_numpy()
        cluster_number = len(only_some_digits)

    # Remove the pixels that only have zeroes in them
    digits = digits.loc[:, (digits != 0).any(axis=0)]
    x = digits.to_numpy()
    n = x.shape[0]

    # Create the weight matrix
    if epsilon is not None:
        weight_matrix = epsilon_ball_sparse(x, epsilon, n=n)
    elif k is not None:
        weight_matrix = knn_sparse(x, k, n, sigma=10.)  # Sigma higher because points further away from each other
    else:
        raise Exception('Either epsilon or k has to be given')

    # Get the graph laplacian
    diagonal = np.squeeze(np.asarray(weight_matrix.sum(axis=0)))
    degree_matrix = sparse.coo_matrix((diagonal, (range(n), range(n))), shape=(n, n))
    laplacian = weight_matrix - degree_matrix

    # Use k-means to get the labels
    vals, vecs = sparse.linalg.eigsh(-laplacian, M=degree_matrix, k=cluster_number, which="SM")
    kmeans = cluster.KMeans(n_clusters=cluster_number).fit(vecs)

    # Plot the spectral embedding (with eigenvectors of -L not -L_sym)
    vals_laplacian, vecs_laplacian = sparse.linalg.eigsh(-laplacian, k=cluster_number, which="SM")
    spectral_embedding = pd.DataFrame(vecs_laplacian)
    scatter_matrix(spectral_embedding, c=kmeans.labels_, s=200)

    # Accuracy of the labels
    accuracy = gl.clustering.clustering_accuracy(kmeans.labels_, labels_original)
    print('Clustering Accuracy: %.2f%%'%accuracy)

    # Check if our graph is connected
    g = ig.Graph.Weighted_Adjacency(weight_matrix.toarray(), mode="undirected")
    print("Graph is connected:", g.is_connected())


#spectral_clustering_for_blobs(k=10, n=1000, centers=[(1, 1), (5, 1)])
#spectral_clustering_for_blobs(epsilon=2, n=1000, centers=[(1, 1), (5, 1)])
#spectral_clustering_for_moons(epsilon=0.4, n=1000)
#spectral_clustering_for_moons(k=40, n=1000)
#spectral_clustering_for_digits(k=20)
#spectral_clustering_for_digits(k=20, only_some_digits=[1, 7])
#spectral_clustering_for_digits(k=20, only_some_digits=[3, 5])

plt.show()
