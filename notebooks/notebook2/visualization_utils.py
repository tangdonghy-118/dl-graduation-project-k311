"""
Visualization utilities for KMeans and GMM demonstrations.

This module contains beautiful and educational plotting functions for
clustering demonstrations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap
from sklearn.metrics import silhouette_samples
import warnings
warnings.filterwarnings('ignore')

# Set beautiful plotting style
plt.style.use('default')
sns.set_palette("husl")


def plot_beautiful_clusters(X, labels, centroids=None, title="Cluster Analysis",
                            figsize=(10, 8), save_path=None):
    """
    Create beautiful cluster visualization.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix (2D for visualization)
    labels : array-like, shape (n_samples,)
        Cluster labels
    centroids : array-like, shape (n_clusters, n_features), optional
        Cluster centroids
    title : str, default="Cluster Analysis"
        Plot title
    figsize : tuple, default=(10, 8)
        Figure size
    save_path : str, optional
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique labels and colors
    unique_labels = np.unique(labels)
    colors = sns.color_palette("husl", len(unique_labels))

    # Plot clusters
    for i, label in enumerate(unique_labels):
        if label == -1:
            # Outliers (for DBSCAN)
            color = 'black'
            marker = 'x'
            alpha = 0.5
        else:
            color = colors[i]
            marker = 'o'
            alpha = 0.7

        mask = labels == label
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=[color], marker=marker, alpha=alpha, s=50,
                   label=f'Cluster {label}' if label != -1 else 'Outliers')

    # Plot centroids
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1],
                   c='red', marker='*', s=200, alpha=0.8,
                   edgecolors='black', linewidth=2, label='Centroids')

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_silhouette_analysis(X, labels, title="Silhouette Analysis", figsize=(12, 8)):
    """
    Create silhouette analysis visualization.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    labels : array-like, shape (n_samples,)
        Cluster labels
    title : str, default="Silhouette Analysis"
        Plot title
    figsize : tuple, default=(12, 8)
        Figure size
    """
    from .clustering_utils import calculate_silhouette_score, calculate_silhouette_samples

    fig, ax1 = plt.subplots(figsize=figsize)

    # Calculate silhouette scores
    silhouette_avg = calculate_silhouette_score(X, labels)
    sample_silhouette_values = calculate_silhouette_samples(X, labels)

    # Plot 1: Silhouette plot
    y_lower = 10
    n_clusters = len(np.unique(labels))

    for i in range(n_clusters):
        # Aggregate silhouette scores for samples in cluster i
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with cluster numbers
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10

    ax1.set_title('Silhouette Plot')
    ax1.set_xlabel('Silhouette Coefficient Values')
    ax1.set_ylabel('Cluster Label')

    # Add vertical line for average silhouette score
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--",
                label=f'Average Score: {silhouette_avg:.3f}')
    ax1.legend()

    # # Plot 2: Actual clusters
    # colors = plt.cm.nipy_spectral(labels.astype(float) / n_clusters)
    # ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
    #             c=colors, edgecolor='k')
    # ax2.set_title('Clustered Data')
    # ax2.set_xlabel('Feature 1')
    # ax2.set_ylabel('Feature 2')

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_elbow_method(k_values, inertias, title="Elbow Method", figsize=(10, 6)):
    """
    Plot elbow method for optimal k selection.

    Parameters:
    -----------
    k_values : list
        List of k values
    inertias : list
        List of corresponding inertia values
    title : str, default="Elbow Method"
        Plot title
    figsize : tuple, default=(10, 6)
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Annotate points
    for i, (k, inertia) in enumerate(zip(k_values, inertias)):
        ax.annotate(f'({k}, {inertia:.0f})',
                    (k, inertia), textcoords="offset points",
                    xytext=(0, 10), ha='center')

    plt.tight_layout()
    plt.show()


def plot_clustering_metrics(metrics_dict, title="Clustering Metrics Comparison",
                            figsize=(12, 8)):
    """
    Plot multiple clustering metrics for comparison.

    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with k values as keys and metrics as values
    title : str, default="Clustering Metrics Comparison"
        Plot title
    figsize : tuple, default=(12, 8)
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    # Extract data
    k_values = list(metrics_dict.keys())

    # Plot different metrics
    metrics = ['silhouette_score', 'davies_bouldin_score',
               'calinski_harabasz_score', 'inertia']

    for i, metric in enumerate(metrics):
        if i < len(axes):
            ax = axes[i]
            values = [metrics_dict[k].get(metric, 0) for k in k_values]

            ax.plot(k_values, values, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} vs k')
            ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_gmm_components(X, gmm_params, title="GMM Components", figsize=(10, 8)):
    """
    Plot GMM components with ellipses.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix (2D for visualization)
    gmm_params : dict
        GMM parameters (weights, means, covariances)
    title : str, default="GMM Components"
        Plot title
    figsize : tuple, default=(10, 8)
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot data points
    labels = gmm_params['labels']
    colors = sns.color_palette("husl", len(np.unique(labels)))

    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=[colors[i]], alpha=0.6, s=50,
                   label=f'Component {label}')

    # Plot ellipses for each component
    for i, (mean, cov, weight) in enumerate(zip(gmm_params['means'],
                                                gmm_params['covariances'],
                                                gmm_params['weights'])):
        # Calculate ellipse parameters
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))

        # Create ellipse
        ellipse = Ellipse(mean, 2 * np.sqrt(eigenvals[0]), 2 * np.sqrt(eigenvals[1]),
                          angle=angle, facecolor=colors[i], alpha=0.3,
                          edgecolor=colors[i], linewidth=2)
        ax.add_patch(ellipse)

        # Plot mean
        ax.scatter(mean[0], mean[1], c='red', marker='*', s=200,
                   edgecolors='black', linewidth=2)

        # Annotate weight
        ax.annotate(f'w={weight:.3f}', mean,
                    xytext=(5, 5), textcoords='offset points')

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_cluster_evolution(history, X, title="Clustering Evolution", figsize=(15, 10)):
    """
    Plot evolution of clustering algorithm.

    Parameters:
    -----------
    history : dict
        History of clustering iterations
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    title : str, default="Clustering Evolution"
        Plot title
    figsize : tuple, default=(15, 10)
        Figure size
    """
    n_iterations = len(history['centroids'])
    cols = min(4, n_iterations)
    rows = (n_iterations + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_iterations):
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        # Get centroids for this iteration
        centroids = history['centroids'][i]

        # Assign points to nearest centroids
        from scipy.spatial.distance import cdist
        distances = cdist(X, centroids)
        labels = np.argmin(distances, axis=1)

        # Plot clusters
        colors = sns.color_palette("husl", len(centroids))
        for j, color in enumerate(colors):
            mask = labels == j
            ax.scatter(X[mask, 0], X[mask, 1],
                       c=[color], alpha=0.6, s=30)

        # Plot centroids
        ax.scatter(centroids[:, 0], centroids[:, 1],
                   c='red', marker='*', s=200,
                   edgecolors='black', linewidth=2)

        ax.set_title(f'Iteration {i+1}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_iterations, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].set_visible(False)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_clustering_comparison(X, results_dict, title="Clustering Comparison",
                               figsize=(15, 10)):
    """
    Compare different clustering results.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    results_dict : dict
        Dictionary with algorithm names as keys and results as values
    title : str, default="Clustering Comparison"
        Plot title
    figsize : tuple, default=(15, 10)
        Figure size
    """
    n_algorithms = len(results_dict)
    cols = min(3, n_algorithms)
    rows = (n_algorithms + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i, (algorithm, result) in enumerate(results_dict.items()):
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        labels = result['labels']
        centroids = result.get('centroids', None)

        # Plot clusters
        unique_labels = np.unique(labels)
        colors = sns.color_palette("husl", len(unique_labels))

        for j, label in enumerate(unique_labels):
            if label == -1:
                color = 'black'
                marker = 'x'
                alpha = 0.5
            else:
                color = colors[j]
                marker = 'o'
                alpha = 0.7

            mask = labels == label
            ax.scatter(X[mask, 0], X[mask, 1],
                       c=[color], marker=marker, alpha=alpha, s=50)

        # Plot centroids
        if centroids is not None:
            ax.scatter(centroids[:, 0], centroids[:, 1],
                       c='red', marker='*', s=200,
                       edgecolors='black', linewidth=2)

        ax.set_title(f'{algorithm}\n'
                     f'Silhouette: {result.get("silhouette_score", 0):.3f}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_algorithms, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].set_visible(False)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def create_clustering_dashboard(X, labels, centroids=None, metrics=None,
                                title="Clustering Dashboard", figsize=(20, 15)):
    """
    Create comprehensive clustering dashboard.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    labels : array-like, shape (n_samples,)
        Cluster labels
    centroids : array-like, optional
        Cluster centroids
    metrics : dict, optional
        Clustering metrics
    title : str, default="Clustering Dashboard"
        Plot title
    figsize : tuple, default=(20, 15)
        Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Main cluster plot
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    unique_labels = np.unique(labels)
    colors = sns.color_palette("husl", len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax1.scatter(X[mask, 0], X[mask, 1],
                    c=[colors[i]], alpha=0.7, s=50,
                    label=f'Cluster {label}')

    if centroids is not None:
        ax1.scatter(centroids[:, 0], centroids[:, 1],
                    c='red', marker='*', s=200,
                    edgecolors='black', linewidth=2, label='Centroids')

    ax1.set_title('Cluster Visualization', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Silhouette analysis
    if len(unique_labels) > 1:
        ax2 = fig.add_subplot(gs[0, 2])
        from .clustering_utils import calculate_silhouette_samples
        sample_silhouette_values = calculate_silhouette_samples(X, labels)

        y_lower = 10
        for i in range(len(unique_labels)):
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            ax2.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=colors[i], alpha=0.7)

            y_lower = y_upper + 10

        ax2.set_title('Silhouette Analysis')
        ax2.set_xlabel('Silhouette Coefficient')

    # Metrics display
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.axis('off')

    if metrics:
        metrics_text = "📊 Clustering Metrics:\n\n"
        for metric, value in metrics.items():
            metrics_text += f"{metric.replace('_', ' ').title()}: {value:.3f}\n"
    else:
        metrics_text = "📊 Clustering Metrics:\n\nNo metrics provided"

    ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    # Cluster characteristics
    ax4 = fig.add_subplot(gs[2, :])
    cluster_sizes = [np.sum(labels == i) for i in unique_labels]
    ax4.bar(range(len(unique_labels)), cluster_sizes, color=colors, alpha=0.7)
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Number of Points')
    ax4.set_title('Cluster Sizes')
    ax4.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_parameter_evolution(history, title="Parameter Evolution", figsize=(15, 10)):
    """
    Plot evolution of clustering parameters.

    Parameters:
    -----------
    history : dict
        History of parameter evolution
    title : str, default="Parameter Evolution"
        Plot title
    figsize : tuple, default=(15, 10)
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    # Plot inertia evolution
    if 'inertias' in history:
        axes[0].plot(history['inertias'], 'o-', linewidth=2)
        axes[0].set_title('Inertia Evolution')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Inertia')
        axes[0].grid(True, alpha=0.3)

    # Plot likelihood evolution (for GMM)
    if 'likelihoods' in history:
        axes[1].plot(history['likelihoods'], 'o-', linewidth=2)
        axes[1].set_title('Likelihood Evolution')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Log-Likelihood')
        axes[1].grid(True, alpha=0.3)

    # Plot centroid movement
    if 'centroids' in history:
        axes[2].set_title('Centroid Movement')
        for i, centroids in enumerate(history['centroids']):
            if i > 0:
                prev_centroids = history['centroids'][i - 1]
                for j in range(len(centroids)):
                    axes[2].arrow(prev_centroids[j, 0], prev_centroids[j, 1],
                                  centroids[j, 0] - prev_centroids[j, 0],
                                  centroids[j, 1] - prev_centroids[j, 1],
                                  head_width=0.1, head_length=0.1,
                                  fc=f'C{j}', ec=f'C{j}', alpha=0.7)
        axes[2].set_xlabel('Feature 1')
        axes[2].set_ylabel('Feature 2')
        axes[2].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_convergence_analysis(history, title="Convergence Analysis", figsize=(12, 8)):
    """
    Plot convergence analysis for clustering algorithms.

    Parameters:
    -----------
    history : dict
        History of algorithm convergence
    title : str, default="Convergence Analysis"
        Plot title
    figsize : tuple, default=(12, 8)
        Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot convergence metric
    if 'inertias' in history:
        axes[0].plot(history['inertias'], 'o-', linewidth=2)
        axes[0].set_title('Inertia Convergence')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Inertia')
        axes[0].grid(True, alpha=0.3)
    elif 'likelihoods' in history:
        axes[0].plot(history['likelihoods'], 'o-', linewidth=2)
        axes[0].set_title('Likelihood Convergence')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Log-Likelihood')
        axes[0].grid(True, alpha=0.3)

    # Plot convergence rate
    if 'inertias' in history and len(history['inertias']) > 1:
        differences = np.diff(history['inertias'])
        axes[1].plot(differences, 'o-', linewidth=2)
        axes[1].set_title('Convergence Rate')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Change in Inertia')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_cluster_characteristics(X, labels, centroids=None,
                                 title="Cluster Characteristics", figsize=(15, 10)):
    """
    Plot detailed cluster characteristics.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    labels : array-like, shape (n_samples,)
        Cluster labels
    centroids : array-like, optional
        Cluster centroids
    title : str, default="Cluster Characteristics"
        Plot title
    figsize : tuple, default=(15, 10)
        Figure size
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    # Cluster sizes
    cluster_sizes = [np.sum(labels == i) for i in unique_labels]
    axes[0].bar(range(n_clusters), cluster_sizes, alpha=0.7)
    axes[0].set_title('Cluster Sizes')
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Number of Points')
    axes[0].grid(True, alpha=0.3)

    # Within-cluster variance
    if centroids is not None:
        variances = []
        for i, label in enumerate(unique_labels):
            cluster_points = X[labels == label]
            if len(cluster_points) > 0:
                variance = np.mean(np.sum((cluster_points - centroids[i])**2, axis=1))
                variances.append(variance)
            else:
                variances.append(0)

        axes[1].bar(range(n_clusters), variances, alpha=0.7)
        axes[1].set_title('Within-Cluster Variance')
        axes[1].set_xlabel('Cluster')
        axes[1].set_ylabel('Variance')
        axes[1].grid(True, alpha=0.3)

    # Feature distributions
    for feature_idx in range(min(2, X.shape[1])):
        ax = axes[2 + feature_idx]

        for i, label in enumerate(unique_labels):
            cluster_points = X[labels == label]
            if len(cluster_points) > 0:
                ax.hist(cluster_points[:, feature_idx],
                        alpha=0.5, label=f'Cluster {label}', bins=20)

        ax.set_title(f'Feature {feature_idx + 1} Distribution')
        ax.set_xlabel(f'Feature {feature_idx + 1} Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_dimensionality_reduction(X, labels, method='PCA',
                                  title="Dimensionality Reduction", figsize=(12, 8)):
    """
    Plot dimensionality reduction visualization.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    labels : array-like, shape (n_samples,)
        Cluster labels
    method : str, default='PCA'
        Dimensionality reduction method
    title : str, default="Dimensionality Reduction"
        Plot title
    figsize : tuple, default=(12, 8)
        Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if method == 'PCA':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        X_reduced = reducer.fit_transform(X)

        # Plot explained variance
        axes[0].bar(range(1, 3), reducer.explained_variance_ratio_)
        axes[0].set_title('PCA Explained Variance')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].grid(True, alpha=0.3)

    elif method == 'TSNE':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        X_reduced = reducer.fit_transform(X)

        axes[0].axis('off')
        axes[0].text(0.5, 0.5, 't-SNE\nNo explained variance',
                     transform=axes[0].transAxes, ha='center', va='center',
                     fontsize=14)

    # Plot reduced dimensions
    colors = sns.color_palette("husl", len(np.unique(labels)))
    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        axes[1].scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                        c=[colors[i]], alpha=0.7, s=50,
                        label=f'Cluster {label}')

    axes[1].set_title(f'{method} Visualization')
    axes[1].set_xlabel('Component 1')
    axes[1].set_ylabel('Component 2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_cluster_validation(X, k_range, results, title="Cluster Validation",
                            figsize=(15, 10)):
    """
    Plot cluster validation metrics.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    k_range : list
        Range of k values tested
    results : dict
        Validation results
    title : str, default="Cluster Validation"
        Plot title
    figsize : tuple, default=(15, 10)
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    metrics = ['silhouette_scores', 'davies_bouldin_scores',
               'calinski_harabasz_scores', 'inertias']

    for i, metric in enumerate(metrics):
        if metric in results:
            axes[i].plot(k_range, results[metric], 'o-', linewidth=2)
            axes[i].set_xlabel('Number of Clusters (k)')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].set_title(f'{metric.replace("_", " ").title()} vs k')
            axes[i].grid(True, alpha=0.3)

            # Highlight optimal k
            if 'optimal_k' in results:
                optimal_k = results['optimal_k']
                if optimal_k in k_range:
                    idx = k_range.index(optimal_k)
                    axes[i].scatter(optimal_k, results[metric][idx],
                                    c='red', s=100, marker='*',
                                    label=f'Optimal k={optimal_k}')
                    axes[i].legend()

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
