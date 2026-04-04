"""
Clustering utilities for KMeans and GMM demonstrations.

This module contains mathematical functions and algorithms for
educational clustering demonstrations, including evaluation metrics,
optimization techniques, and model comparison utilities.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score, silhouette_samples, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score, adjusted_mutual_info_score,
    completeness_score, homogeneity_score, v_measure_score
)
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')


def calculate_silhouette_score(X, labels, metric='euclidean', sample_size=None):
    """
    Calculate silhouette score for cluster evaluation.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    labels : array-like, shape (n_samples,)
        Cluster labels
    metric : str, default='euclidean'
        Distance metric
    sample_size : int, default=None
        Sample size for computation

    Returns:
    --------
    float
        Silhouette score
    """
    try:
        if len(set(labels)) < 2:
            return 0.0
        return silhouette_score(X, labels, metric=metric, sample_size=sample_size)
    except Exception as e:
        print(f"Error calculating silhouette score: {e}")
        return 0.0


def calculate_silhouette_samples(X, labels, metric='euclidean'):
    """
    Calculate silhouette score for each sample.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    labels : array-like, shape (n_samples,)
        Cluster labels
    metric : str, default='euclidean'
        Distance metric

    Returns:
    --------
    array-like, shape (n_samples,)
        Silhouette scores for each sample
    """
    try:
        if len(set(labels)) < 2:
            return np.zeros(len(X))
        return silhouette_samples(X, labels, metric=metric)
    except Exception as e:
        print(f"Error calculating silhouette samples: {e}")
        return np.zeros(len(X))


def calculate_davies_bouldin_score(X, labels):
    """
    Calculate Davies-Bouldin score for cluster evaluation.
    Lower values indicate better clustering.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    labels : array-like, shape (n_samples,)
        Cluster labels

    Returns:
    --------
    float
        Davies-Bouldin score
    """
    try:
        if len(set(labels)) < 2:
            return float('inf')
        return davies_bouldin_score(X, labels)
    except Exception as e:
        print(f"Error calculating Davies-Bouldin score: {e}")
        return float('inf')


def calculate_calinski_harabasz_score(X, labels):
    """
    Calculate Calinski-Harabasz score for cluster evaluation.
    Higher values indicate better clustering.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    labels : array-like, shape (n_samples,)
        Cluster labels

    Returns:
    --------
    float
        Calinski-Harabasz score
    """
    try:
        if len(set(labels)) < 2:
            return 0.0
        return calinski_harabasz_score(X, labels)
    except Exception as e:
        print(f"Error calculating Calinski-Harabasz score: {e}")
        return 0.0


def calculate_inertia(X, labels, centroids):
    """
    Calculate within-cluster sum of squares (inertia).

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    labels : array-like, shape (n_samples,)
        Cluster labels
    centroids : array-like, shape (n_clusters, n_features)
        Cluster centroids

    Returns:
    --------
    float
        Inertia value
    """
    try:
        inertia = 0
        for i in range(len(centroids)):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                distances = np.sum((cluster_points - centroids[i]) ** 2, axis=1)
                inertia += np.sum(distances)
        return inertia
    except Exception as e:
        print(f"Error calculating inertia: {e}")
        return float('inf')


def calculate_elbow_method(X, k_range=range(1, 11), random_state=42):
    """
    Calculate inertia values for different k values (elbow method).

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    k_range : range, default=range(1, 11)
        Range of k values to test
    random_state : int, default=42
        Random seed

    Returns:
    --------
    dict
        Dictionary with k values and corresponding inertia values
    """
    inertias = []
    k_values = list(k_range)

    X_array = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)

    for k in k_values:
        if k == 1:
            centroid = X_array.mean(axis=0)
            inertia = ((X_array - centroid) ** 2).sum()
        else:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            kmeans.fit(X_array)
            inertia = kmeans.inertia_

        inertias.append(float(inertia))

    return {
        "k_values": k_values,
        "inertias": inertias
    }

def calculate_aic_bic(X, k_range=range(1, 11), random_state=42):
    """
    Calculate AIC and BIC for GMM model selection.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    k_range : range, default=range(1, 11)
        Range of k values to test
    random_state : int, default=42
        Random seed

    Returns:
    --------
    dict
        Dictionary with k values, AIC, and BIC values
    """
    aic_values = []
    bic_values = []
    k_values = list(k_range)

    for k in k_values:
        try:
            gmm = GaussianMixture(n_components=k, random_state=random_state, covariance_type= 'diag')
            gmm.fit(X)

            aic_values.append(gmm.aic(X))
            bic_values.append(gmm.bic(X))
        except Exception as e:
            print(f"Error fitting GMM with k={k}: {e}")
            aic_values.append(float('inf'))
            bic_values.append(float('inf'))

    return {
        'k_values': k_values,
        'aic_values': aic_values,
        'bic_values': bic_values
    }


def find_optimal_clusters(X, k_range=range(2, 11), methods=['silhouette', 'elbow'],
                          random_state=42):
    """
    Find optimal number of clusters using multiple methods.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    k_range : range, default=range(2, 11)
        Range of k values to test
    methods : list, default=['silhouette', 'elbow']
        Methods to use for optimization
    random_state : int, default=42
        Random seed

    Returns:
    --------
    dict
        Dictionary with optimization results
    """
    results = {}

    if 'silhouette' in methods:
        silhouette_scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            score = calculate_silhouette_score(X, labels)
            silhouette_scores.append(score)

        results['silhouette'] = {
            'k_values': list(k_range),
            'scores': silhouette_scores,
            'optimal_k': list(k_range)[np.argmax(silhouette_scores)]
        }

    if 'elbow' in methods:
        elbow_results = calculate_elbow_method(X, k_range, random_state)
        # Find elbow point using the "elbow" heuristic
        inertias = elbow_results['inertias']
        k_values = elbow_results['k_values']

        # Calculate second derivative to find elbow
        if len(inertias) >= 3:
            second_derivative = []
            for i in range(1, len(inertias) - 1):
                second_deriv = inertias[i - 1] - 2 * inertias[i] + inertias[i + 1]
                second_derivative.append(second_deriv)

            if second_derivative:
                elbow_idx = np.argmax(second_derivative) + 1
                optimal_k = k_values[elbow_idx]
            else:
                optimal_k = k_values[len(k_values) // 2]
        else:
            optimal_k = k_values[len(k_values) // 2]

        results['elbow'] = {
            'k_values': k_values,
            'inertias': inertias,
            'optimal_k': optimal_k
        }

    return results


def evaluate_clustering_performance(X, labels, true_labels=None):
    """
    Comprehensive clustering performance evaluation.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    labels : array-like, shape (n_samples,)
        Predicted cluster labels
    true_labels : array-like, shape (n_samples,), optional
        True cluster labels for external validation

    Returns:
    --------
    dict
        Dictionary with performance metrics
    """
    metrics = {}

    # Internal metrics
    metrics['silhouette_score'] = calculate_silhouette_score(X, labels)
    metrics['davies_bouldin_score'] = calculate_davies_bouldin_score(X, labels)
    metrics['calinski_harabasz_score'] = calculate_calinski_harabasz_score(X, labels)

    # External metrics (if true labels available)
    if true_labels is not None:
        try:
            metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, labels)
            metrics['adjusted_mutual_info_score'] = adjusted_mutual_info_score(true_labels, labels)
            metrics['homogeneity_score'] = homogeneity_score(true_labels, labels)
            metrics['completeness_score'] = completeness_score(true_labels, labels)
            metrics['v_measure_score'] = v_measure_score(true_labels, labels)
        except Exception as e:
            print(f"Error calculating external metrics: {e}")

    return metrics


def create_synthetic_clustering_data(n_samples=300, n_features=2, n_centers=3,
                                     cluster_std=1.0, center_box=(-10, 10),
                                     random_state=42):
    """
    Create synthetic clustering data for demonstrations.

    Parameters:
    -----------
    n_samples : int, default=300
        Number of samples
    n_features : int, default=2
        Number of features
    n_centers : int, default=3
        Number of cluster centers
    cluster_std : float, default=1.0
        Standard deviation of clusters
    center_box : tuple, default=(-10, 10)
        Bounding box for cluster centers
    random_state : int, default=42
        Random seed

    Returns:
    --------
    tuple
        (X, y) where X is feature matrix and y is true labels
    """
    return make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_centers,
        cluster_std=cluster_std,
        center_box=center_box,
        shuffle=True,
        random_state=random_state
    )


def initialize_centroids(X, k, method='k-means++', random_state=42):
    """
    Initialize cluster centroids using different methods.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    k : int
        Number of clusters
    method : str, default='k-means++'
        Initialization method
    random_state : int, default=42
        Random seed

    Returns:
    --------
    array-like, shape (k, n_features)
        Initial centroids
    """
    np.random.seed(random_state)

    if method == 'random':
        # Random initialization
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        centroids = np.random.uniform(min_vals, max_vals, (k, X.shape[1]))

    elif method == 'k-means++':
        # K-means++ initialization
        centroids = []

        # Choose first centroid randomly
        first_centroid = X[np.random.choice(X.shape[0])]
        centroids.append(first_centroid)

        # Choose remaining centroids
        for _ in range(k - 1):
            # Calculate distances to nearest centroid
            distances = []
            for point in X:
                min_dist = float('inf')
                for centroid in centroids:
                    dist = np.linalg.norm(point - centroid)
                    min_dist = min(min_dist, dist)
                distances.append(min_dist)

            # Choose next centroid with probability proportional to squared distance
            distances = np.array(distances)
            probabilities = distances ** 2
            probabilities = probabilities / np.sum(probabilities)

            next_centroid_idx = np.random.choice(X.shape[0], p=probabilities)
            centroids.append(X[next_centroid_idx])

        centroids = np.array(centroids)

    else:
        raise ValueError(f"Unknown initialization method: {method}")

    return centroids


def kmeans_iteration(X, centroids, max_iters=100, tol=1e-4):
    """
    Perform K-means clustering iterations.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    centroids : array-like, shape (k, n_features)
        Initial centroids
    max_iters : int, default=100
        Maximum number of iterations
    tol : float, default=1e-4
        Convergence tolerance

    Returns:
    --------
    tuple
        (final_centroids, labels, history)
    """
    history = {'centroids': [centroids.copy()], 'inertias': []}

    for iteration in range(max_iters):
        # Assign points to nearest centroids
        distances = cdist(X, centroids)
        labels = np.argmin(distances, axis=1)

        # Calculate inertia
        inertia = 0
        for i in range(len(centroids)):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[i]) ** 2)
        history['inertias'].append(inertia)

        # Update centroids
        new_centroids = []
        for i in range(len(centroids)):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroid = np.mean(cluster_points, axis=0)
            else:
                new_centroid = centroids[i]  # Keep old centroid if no points
            new_centroids.append(new_centroid)

        new_centroids = np.array(new_centroids)

        # Check convergence
        centroid_shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
        if centroid_shift < tol:
            break

        centroids = new_centroids
        history['centroids'].append(centroids.copy())

    return centroids, labels, history


def calculate_gmm_likelihood(X, weights, means, covariances):
    """
    Calculate likelihood for Gaussian Mixture Model.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    weights : array-like, shape (n_components,)
        Component weights
    means : array-like, shape (n_components, n_features)
        Component means
    covariances : array-like, shape (n_components, n_features, n_features)
        Component covariances

    Returns:
    --------
    float
        Log-likelihood
    """
    from scipy.stats import multivariate_normal

    n_samples, n_features = X.shape
    n_components = len(weights)

    likelihood = 0
    for i in range(n_samples):
        point_likelihood = 0
        for j in range(n_components):
            try:
                component_likelihood = multivariate_normal.pdf(
                    X[i], means[j], covariances[j]
                )
                point_likelihood += weights[j] * component_likelihood
            except BaseException:
                # Handle numerical issues
                point_likelihood += weights[j] * 1e-10

        likelihood += np.log(max(point_likelihood, 1e-10))

    return likelihood


def gaussian_mixture_em_step(X, weights, means, covariances, max_iters=100, tol=1e-6):
    """
    Perform EM algorithm for Gaussian Mixture Model.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    weights : array-like, shape (n_components,)
        Initial component weights
    means : array-like, shape (n_components, n_features)
        Initial component means
    covariances : array-like, shape (n_components, n_features, n_features)
        Initial component covariances
    max_iters : int, default=100
        Maximum number of iterations
    tol : float, default=1e-6
        Convergence tolerance

    Returns:
    --------
    tuple
        (final_weights, final_means, final_covariances, history)
    """
    from scipy.stats import multivariate_normal

    n_samples, n_features = X.shape
    n_components = len(weights)

    history = {
        'weights': [weights.copy()],
        'means': [means.copy()],
        'covariances': [covariances.copy()],
        'likelihoods': []
    }

    for iteration in range(max_iters):
        # E-step: Calculate responsibilities
        responsibilities = np.zeros((n_samples, n_components))

        for i in range(n_samples):
            for j in range(n_components):
                try:
                    responsibilities[i, j] = weights[j] * multivariate_normal.pdf(
                        X[i], means[j], covariances[j]
                    )
                except BaseException:
                    responsibilities[i, j] = weights[j] * 1e-10

        # Normalize responsibilities
        row_sums = responsibilities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        responsibilities = responsibilities / row_sums

        # M-step: Update parameters
        Nk = responsibilities.sum(axis=0)

        # Update weights
        new_weights = Nk / n_samples

        # Update means
        new_means = np.zeros((n_components, n_features))
        for j in range(n_components):
            if Nk[j] > 0:
                new_means[j] = np.sum(responsibilities[:, j].reshape(-1, 1) * X, axis=0) / Nk[j]
            else:
                new_means[j] = means[j]

        # Update covariances
        new_covariances = np.zeros((n_components, n_features, n_features))
        for j in range(n_components):
            if Nk[j] > 0:
                diff = X - new_means[j]
                new_covariances[j] = np.dot(
                    (responsibilities[:, j].reshape(-1, 1) * diff).T, diff
                ) / Nk[j]

                # Add small value to diagonal for numerical stability
                new_covariances[j] += np.eye(n_features) * 1e-6
            else:
                new_covariances[j] = covariances[j]

        # Calculate likelihood
        likelihood = calculate_gmm_likelihood(X, new_weights, new_means, new_covariances)
        history['likelihoods'].append(likelihood)

        # Check convergence
        if iteration > 0:
            if abs(likelihood - history['likelihoods'][-2]) < tol:
                break

        # Update parameters
        weights = new_weights
        means = new_means
        covariances = new_covariances

        history['weights'].append(weights.copy())
        history['means'].append(means.copy())
        history['covariances'].append(covariances.copy())

    return weights, means, covariances, history


def calculate_gmm_parameters(X, n_components, random_state=42):
    """
    Calculate GMM parameters using sklearn for comparison.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    n_components : int
        Number of components
    random_state : int, default=42
        Random seed

    Returns:
    --------
    dict
        Dictionary with GMM parameters
    """
    try:
        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        gmm.fit(X)

        return {
            'weights': gmm.weights_,
            'means': gmm.means_,
            'covariances': gmm.covariances_,
            'aic': gmm.aic(X),
            'bic': gmm.bic(X),
            'labels': gmm.predict(X),
            'probabilities': gmm.predict_proba(X)
        }
    except Exception as e:
        print(f"Error calculating GMM parameters: {e}")
        return None
