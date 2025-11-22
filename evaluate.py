import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from kmeans import KMeans
import os

def generate_dataset(n_samples=500, n_clusters=3, random_state=42):
    """
    Generate synthetic 2D dataset with well-defined, non-linearly separable clusters.
    
    Parameters:
    -----------
    n_samples : int, default=500
        Total number of samples.
    n_clusters : int, default=3
        Number of clusters.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Returns:
    --------
    X : array, shape (n_samples, 2)
        Generated samples.
    true_labels : array, shape (n_samples,)
        True cluster labels.
    """
    X, true_labels = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=n_clusters,
        cluster_std=1.5,
        center_box=(-10.0, 10.0),
        shuffle=True,
        random_state=random_state
    )
    return X, true_labels

def plot_clusters(X, labels, centroids, title, filename):
    """
    Create scatter plot visualization of clusters.
    
    Parameters:
    -----------
    X : array, shape (n_samples, 2)
        Data points.
    labels : array, shape (n_samples,)
        Cluster labels.
    centroids : array, shape (n_clusters, 2)
        Centroid positions.
    title : str
        Plot title.
    filename : str
        Filename to save the plot.
    """
    plt.figure(figsize=(10, 8))
    
    # Plot data points colored by cluster
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                         alpha=0.6, edgecolors='k', s=50)
    
    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], 
               c='red', marker='X', s=300, edgecolors='k', 
               linewidths=2, label='Centroids')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.legend(fontsize=12)
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig(f'visualizations/{filename}', dpi=300, bbox_inches='tight')
    print(f"Saved visualization: visualizations/{filename}")
    plt.close()

def evaluate_initialization(X, init_method, n_clusters=3, random_state=42):
    """
    Evaluate K-Means with a specific initialization method.
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
        Training data.
    init_method : str
        Initialization method: 'random' or 'kmeans++'
    n_clusters : int, default=3
        Number of clusters.
    random_state : int, default=42
        Random seed.
        
    Returns:
    --------
    kmeans : KMeans
        Fitted K-Means model.
    """
    kmeans = KMeans(n_clusters=n_clusters, max_iters=300, 
                   tol=1e-4, random_state=random_state)
    kmeans.fit(X, init=init_method)
    return kmeans

def print_results(method_name, kmeans):
    """
    Print evaluation results.
    
    Parameters:
    -----------
    method_name : str
        Name of the initialization method.
    kmeans : KMeans
        Fitted K-Means model.
    """
    print(f"\n{'='*60}")
    print(f"Results for {method_name} Initialization")
    print(f"{'='*60}")
    print(f"Number of iterations: {kmeans.n_iter}")
    print(f"Final SSE (Inertia): {kmeans.inertia:.4f}")
    print(f"Centroid positions:")
    for i, centroid in enumerate(kmeans.centroids):
        print(f"  Cluster {i}: [{centroid[0]:.4f}, {centroid[1]:.4f}]")
    print(f"{'='*60}")

def main():
    """
    Main evaluation script.
    """
    print("K-Means Clustering Evaluation")
    print("==============================\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Step 1: Generate synthetic dataset
    print("Step 1: Generating synthetic dataset...")
    X, true_labels = generate_dataset(n_samples=500, n_clusters=3, random_state=42)
    print(f"Generated {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Data range: X1=[{X[:, 0].min():.2f}, {X[:, 0].max():.2f}], "
          f"X2=[{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
    
    # Step 2: Evaluate with random initialization
    print("\nStep 2: Running K-Means with Random Initialization...")
    kmeans_random = evaluate_initialization(X, init_method='random', 
                                           n_clusters=3, random_state=42)
    print_results("Random", kmeans_random)
    
    # Step 3: Evaluate with K-Means++ initialization
    print("\nStep 3: Running K-Means with K-Means++ Initialization...")
    kmeans_pp = evaluate_initialization(X, init_method='kmeans++', 
                                       n_clusters=3, random_state=42)
    print_results("K-Means++", kmeans_pp)
    
    # Step 4: Generate visualizations
    print("\nStep 4: Generating visualizations...")
    plot_clusters(X, kmeans_random.labels, kmeans_random.centroids,
                 'K-Means Clustering - Random Initialization',
                 'kmeans_random_init.png')
    
    plot_clusters(X, kmeans_pp.labels, kmeans_pp.centroids,
                 'K-Means Clustering - K-Means++ Initialization',
                 'kmeans_plusplus_init.png')
    
    # Step 5: Comparative analysis
    print("\nStep 5: Comparative Analysis")
    print("="*60)
    print(f"SSE Comparison:")
    print(f"  Random Initialization: {kmeans_random.inertia:.4f}")
    print(f"  K-Means++ Initialization: {kmeans_pp.inertia:.4f}")
    
    sse_diff = kmeans_random.inertia - kmeans_pp.inertia
    sse_diff_pct = (sse_diff / kmeans_random.inertia) * 100
    
    if sse_diff > 0:
        print(f"  K-Means++ achieved {sse_diff:.4f} lower SSE ({sse_diff_pct:.2f}% improvement)")
    else:
        print(f"  Random achieved {abs(sse_diff):.4f} lower SSE ({abs(sse_diff_pct):.2f}% improvement)")
    
    print(f"\nIteration Comparison:")
    print(f"  Random Initialization: {kmeans_random.n_iter} iterations")
    print(f"  K-Means++ Initialization: {kmeans_pp.n_iter} iterations")
    
    iter_diff = kmeans_random.n_iter - kmeans_pp.n_iter
    if iter_diff > 0:
        print(f"  K-Means++ converged {iter_diff} iterations faster")
    else:
        print(f"  Random converged {abs(iter_diff)} iterations faster")
    
    print("="*60)
    
    # Step 6: Save results to text file
    print("\nStep 6: Saving results to file...")
    with open('results.txt', 'w') as f:
        f.write("K-Means Clustering Evaluation Results\n")
        f.write("="*60 + "\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"  Samples: {X.shape[0]}\n")
        f.write(f"  Features: {X.shape[1]}\n")
        f.write(f"  Clusters: 3\n\n")
        
        f.write("Random Initialization Results:\n")
        f.write(f"  Iterations: {kmeans_random.n_iter}\n")
        f.write(f"  SSE (Inertia): {kmeans_random.inertia:.4f}\n")
        f.write(f"  Centroids:\n")
        for i, centroid in enumerate(kmeans_random.centroids):
            f.write(f"    Cluster {i}: [{centroid[0]:.4f}, {centroid[1]:.4f}]\n")
        f.write("\n")
        
        f.write("K-Means++ Initialization Results:\n")
        f.write(f"  Iterations: {kmeans_pp.n_iter}\n")
        f.write(f"  SSE (Inertia): {kmeans_pp.inertia:.4f}\n")
        f.write(f"  Centroids:\n")
        for i, centroid in enumerate(kmeans_pp.centroids):
            f.write(f"    Cluster {i}: [{centroid[0]:.4f}, {centroid[1]:.4f}]\n")
        f.write("\n")
        
        f.write("Comparative Analysis:\n")
        f.write(f"  SSE Difference: {sse_diff:.4f} ({sse_diff_pct:.2f}%)\n")
        f.write(f"  Iteration Difference: {iter_diff} iterations\n")
        
        if kmeans_pp.inertia < kmeans_random.inertia:
            f.write("\n  Conclusion: K-Means++ initialization achieved better clustering ")
            f.write("with lower SSE.\n")
        else:
            f.write("\n  Conclusion: Random initialization achieved better clustering ")
            f.write("with lower SSE.\n")
    
    print("Results saved to results.txt")
    
    print("\nEvaluation completed successfully!")
    print("Check the 'visualizations' folder for cluster plots.")

if __name__ == "__main__":
    main()
