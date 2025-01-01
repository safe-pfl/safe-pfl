import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
)
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.preprocessing import normalize


def normalize_matrix(matrix, method="min-max"):
    """
    Normalize a distance or similarity matrix.
    Args:
        matrix (numpy array): The matrix to normalize.
        method (str): Normalization method, "min-max" or "z-score".
    Returns:
        numpy array: The normalized matrix.
    """
    if method == "min-max":
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        normalized_matrix = (matrix - min_val) / (max_val - min_val)
    elif method == "z-score":
        mean_val = np.mean(matrix)
        std_val = np.std(matrix)
        normalized_matrix = (matrix - mean_val) / std_val
    else:
        raise ValueError(
            "Unsupported normalization method. Use 'min-max' or 'z-score'."
        )
    return normalized_matrix


# Load and Normalize Matrices
def load_and_normalize_matrix(filename, method="min-max"):
    """
    Load a matrix from a CSV file and normalize it.
    Args:
        filename (str): Path to the CSV file.
        method (str): Normalization method, "min-max" or "z-score".
    Returns:
        numpy array: The normalized matrix.
    """
    df = pd.read_csv(filename, index_col=0)  # Assuming the first column is an index
    matrix = df.values
    return normalize_matrix(matrix, method=method)


# Perform Clustering with Affinity Propagation
def perform_affinity_propagation(similarity_matrix):
    """
    Perform clustering using Affinity Propagation on a similarity matrix.
    Args:
        similarity_matrix: A precomputed similarity matrix.
    Returns:
        Cluster labels for each client.
    """
    clustering = AffinityPropagation(affinity="precomputed")
    clustering.fit(similarity_matrix)
    return clustering.labels_


# Perform Means-Based Clustering (K-Means)
def perform_means_clustering(distance_matrix, n_clusters):
    """
    Perform clustering using K-Means on a distance matrix.
    Args:
        distance_matrix: A precomputed distance matrix.
        n_clusters: Number of clusters to form.
    Returns:
        Cluster labels for each client.
    """
    # Convert distance matrix to feature space (embedding)
    # Normalize the distance matrix rows to unit vectors for K-Means
    features = normalize(distance_matrix, axis=1, norm="l2")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    print(kmeans.labels_)
    return kmeans.labels_


# Evaluate Clustering
def evaluate_clustering(predicted_labels, ground_truth_labels):
    """
    Evaluate clustering results using ARI, NMI, and FMI.
    Args:
        predicted_labels: Predicted cluster labels for each client.
        ground_truth_labels: Ground truth cluster labels for each client.
    Returns:
        ARI, NMI, FMI scores.
    """
    ari = adjusted_rand_score(ground_truth_labels, predicted_labels)
    nmi = normalized_mutual_info_score(ground_truth_labels, predicted_labels)
    fmi = fowlkes_mallows_score(ground_truth_labels, predicted_labels)
    return ari, nmi, fmi


# Main Workflow
def main():
    # Load ground truth similarity matrix
    ground_truth_matrix = load_and_normalize_matrix(
        "clients_datasets_similarity_matrix.csv", method="min-max"
    )

    # Load other distance/similarity matrices
    coordinate_matrix = load_and_normalize_matrix(
        "distances_coordinate.csv", method="min-max"
    )
    cosine_matrix = load_and_normalize_matrix("distances_cosine.csv", method="min-max")
    euclidean_matrix = load_and_normalize_matrix(
        "distances_euclidean.csv", method="min-max"
    )
    jensen_shannon_matrix = load_and_normalize_matrix(
        "distances_jensen-shannon.csv", method="min-max"
    )
    wasserstein_matrix = load_and_normalize_matrix(
        "distances_wasserstein.csv", method="min-max"
    )

    # Perform clustering for each matrix
    ground_truth_labels = perform_means_clustering(ground_truth_matrix, 3)
    coordinate_labels = perform_means_clustering(coordinate_matrix, 3)
    cosine_labels = perform_means_clustering(cosine_matrix, 3)
    euclidean_labels = perform_means_clustering(euclidean_matrix, 3)
    jensen_shannon_labels = perform_means_clustering(jensen_shannon_matrix, 3)
    wasserstein_labels = perform_means_clustering(wasserstein_matrix, 3)

    # ground_truth_labels = perform_affinity_propagation(ground_truth_matrix)
    # coordinate_labels = perform_affinity_propagation(coordinate_matrix)
    # cosine_labels = perform_affinity_propagation(cosine_matrix)
    # euclidean_labels = perform_affinity_propagation(euclidean_matrix)
    # jensen_shannon_labels = perform_affinity_propagation(jensen_shannon_matrix)
    # wasserstein_labels = perform_affinity_propagation(wasserstein_matrix)

    # Evaluate clustering accuracy
    metrics = {
        "Coordinate": coordinate_labels,
        "Cosine": cosine_labels,
        "Euclidean": euclidean_labels,
        "Jensen-Shannon": jensen_shannon_labels,
        "Wasserstein": wasserstein_labels,
    }

    print("Clustering Evaluation Results:")
    for metric, labels in metrics.items():
        ari, nmi, fmi = evaluate_clustering(labels, ground_truth_labels)
        print(f"{metric} Clustering:")
        print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
        print(f"  Normalized Mutual Information (NMI): {nmi:.4f}")
        print(f"  Fowlkes-Mallows Index (FMI): {fmi:.4f}")
        print()


if __name__ == "__main__":
    main()
