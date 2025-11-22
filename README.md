# K-Means Clustering from Scratch

Implementation of K-Means clustering algorithm from scratch using only NumPy with evaluation of different initialization strategies.

## Overview

This project implements the K-Means clustering algorithm from scratch without using scikit-learn's KMeans class for the core logic. The implementation includes:

- Custom K-Means class with complete clustering pipeline
- Random initialization strategy
- K-Means++ inspired initialization strategy
- Euclidean distance metric
- Convergence criteria based on centroid change
- SSE (Sum of Squared Errors) / Inertia computation
- Comprehensive evaluation and visualization tools

## Project Structure

```
kmeans-clustering-from-scratch/
├── kmeans.py           # K-Means algorithm implementation
├── evaluate.py         # Evaluation script with visualizations
├── requirements.txt    # Project dependencies
├── README.md          # This file
└── .gitignore         # Git ignore file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/joshnanarendran/kmeans-clustering-from-scratch.git
cd kmeans-clustering-from-scratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Evaluation Script

The evaluation script generates a synthetic dataset and compares both initialization strategies:

```bash
python evaluate.py
```

This will:
- Generate a synthetic 2D dataset with 500 points and 3 clusters
- Run K-Means with random initialization
- Run K-Means with K-Means++ initialization
- Generate visualizations in the `visualizations/` folder
- Save detailed results to `results.txt`

### Using the K-Means Class

```python
from kmeans import KMeans
import numpy as np

# Generate sample data
X = np.random.randn(100, 2)

# Initialize K-Means
kmeans = KMeans(n_clusters=3, max_iters=300, tol=1e-4, random_state=42)

# Fit with random initialization
kmeans.fit(X, init='random')

# Or fit with K-Means++ initialization
kmeans.fit(X, init='kmeans++')

# Access results
print(f"Final SSE: {kmeans.inertia}")
print(f"Number of iterations: {kmeans.n_iter}")
print(f"Cluster labels: {kmeans.labels}")
print(f"Centroids: {kmeans.centroids}")
```

## Implementation Details

### K-Means Algorithm

The K-Means implementation follows the standard iterative approach:

1. **Initialization**: Initialize k centroids using either:
   - Random selection: Randomly select k data points as initial centroids
   - K-Means++: Probabilistically select centroids to maximize initial spread

2. **Assignment Step**: Assign each data point to the nearest centroid using Euclidean distance

3. **Update Step**: Recompute centroids as the mean of all points assigned to each cluster

4. **Convergence Check**: Repeat steps 2-3 until:
   - Centroid change is below tolerance threshold (default: 1e-4)
   - OR maximum iterations reached (default: 300)

### Distance Metric

The implementation uses Euclidean distance:

```
d(x, y) = sqrt(sum((x_i - y_i)^2))
```

### Convergence Criteria

Convergence is determined by the total change in centroid positions:

```
convergence = sqrt(sum((new_centroids - old_centroids)^2)) < tolerance
```

### Initialization Strategies

#### Random Initialization
- Randomly selects k data points from the dataset as initial centroids
- Simple but may lead to suboptimal clustering
- Faster initialization

#### K-Means++ Initialization
- First centroid chosen randomly
- Subsequent centroids chosen with probability proportional to squared distance from nearest existing centroid
- Better initial spread leads to faster convergence and better results
- Slightly slower initialization but typically better overall performance

## Dataset

The evaluation uses `sklearn.datasets.make_blobs` to generate a synthetic 2D dataset:

- **Samples**: 500 points
- **Features**: 2 dimensions
- **Clusters**: 3 well-defined, non-linearly separable clusters
- **Cluster Standard Deviation**: 1.5
- **Center Box**: (-10.0, 10.0)

This dataset tests the algorithm's ability to identify distinct clusters in a controlled environment.

## Results

The evaluation script generates:

1. **Console Output**: Detailed metrics for both initialization methods
2. **Visualizations**: Scatter plots showing:
   - Data points colored by assigned cluster
   - Final centroid positions marked with red X markers
3. **results.txt**: Comprehensive comparison including:
   - SSE/Inertia values
   - Number of iterations
   - Centroid positions
   - Performance comparison

## Key Features

- **Pure NumPy Implementation**: Core K-Means logic uses only NumPy
- **Multiple Initialization Strategies**: Compare random vs K-Means++
- **Comprehensive Evaluation**: Automatic comparison and visualization
- **Well-Documented Code**: Clear docstrings and comments
- **Reproducible Results**: Fixed random seeds for consistency
- **Visualization Support**: Automatic generation of cluster plots

## Dependencies

- `numpy>=1.21.0`: Numerical computations and array operations
- `scikit-learn>=1.0.0`: Dataset generation (make_blobs)
- `matplotlib>=3.4.0`: Visualization and plotting

## License

This project is available for educational purposes.

## Author

joshnanarendran

## Acknowledgments

This project was created as part of a machine learning course to demonstrate understanding of:
- Unsupervised learning algorithms
- Iterative optimization processes
- Distance metrics and initialization strategies
- Algorithm evaluation and comparison
