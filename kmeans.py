import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=300, tol=1e-4, random_state=None):
        """
        K-Means clustering implementation from scratch using NumPy.
        
        Parameters:
        -----------
        n_clusters : int, default=3
            The number of clusters to form.
        max_iters : int, default=300
            Maximum number of iterations of the k-means algorithm.
        tol : float, default=1e-4
            Relative tolerance with regards to centroid change to declare convergence.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.n_iter = 0
        
    def _initialize_centroids_random(self, X):
        """
        Initialize centroids by randomly selecting k data points.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
            
        Returns:
        --------
        centroids : array, shape (n_clusters, n_features)
            Initial centroids.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples = X.shape[0]
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        centroids = X[random_indices]
        return centroids
    
    def _initialize_centroids_kmeans_plus_plus(self, X):
        """
        Initialize centroids using K-Means++ strategy.
        The first centroid is chosen randomly, then each subsequent centroid
        is chosen from remaining data points with probability proportional
        to squared distance from nearest existing centroid.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
            
        Returns:
        --------
        centroids : array, shape (n_clusters, n_features)
            Initial centroids.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        
        # Choose first centroid randomly
        first_idx = np.random.randint(0, n_samples)
        centroids[0] = X[first_idx]
        
        # Choose remaining centroids
        for k in range(1, self.n_clusters):
            # Compute squared distances to nearest centroid
            distances = np.zeros(n_samples)
            for i in range(n_samples):
                # Distance to nearest centroid so far
                min_dist = np.inf
                for j in range(k):
                    dist = np.sum((X[i] - centroids[j]) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                distances[i] = min_dist
            
            # Choose next centroid with probability proportional to squared distance
            probabilities = distances / np.sum(distances)
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.rand()
            for i, cum_prob in enumerate(cumulative_probs):
                if r < cum_prob:
                    centroids[k] = X[i]
                    break
        
        return centroids
    
    def _compute_distances(self, X, centroids):
        """
        Compute Euclidean distances between each data point and each centroid.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        centroids : array, shape (n_clusters, n_features)
            Current centroids.
            
        Returns:
        --------
        distances : array, shape (n_samples, n_clusters)
            Distance matrix.
        """
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        
        for k in range(self.n_clusters):
            # Euclidean distance
            distances[:, k] = np.sqrt(np.sum((X - centroids[k]) ** 2, axis=1))
        
        return distances
    
    def _assign_clusters(self, X, centroids):
        """
        Assign each data point to the nearest centroid.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        centroids : array, shape (n_clusters, n_features)
            Current centroids.
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster labels for each data point.
        """
        distances = self._compute_distances(X, centroids)
        labels = np.argmin(distances, axis=1)
        return labels
    
    def _update_centroids(self, X, labels):
        """
        Update centroids by computing mean of all points assigned to each cluster.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        labels : array, shape (n_samples,)
            Current cluster labels.
            
        Returns:
        --------
        centroids : array, shape (n_clusters, n_features)
            Updated centroids.
        """
        n_features = X.shape[1]
        centroids = np.zeros((self.n_clusters, n_features))
        
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # If cluster is empty, reinitialize it randomly
                centroids[k] = X[np.random.randint(0, X.shape[0])]
        
        return centroids
    
    def _compute_inertia(self, X, labels, centroids):
        """
        Compute sum of squared errors (SSE) or inertia.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        labels : array, shape (n_samples,)
            Cluster labels.
        centroids : array, shape (n_clusters, n_features)
            Centroids.
            
        Returns:
        --------
        inertia : float
            Sum of squared distances to nearest centroid.
        """
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[k]) ** 2)
        return inertia
    
    def fit(self, X, init='random'):
        """
        Fit K-Means clustering model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        init : str, default='random'
            Initialization method: 'random' or 'kmeans++'
            
        Returns:
        --------
        self : object
            Fitted estimator.
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Initialize centroids
        if init == 'kmeans++':
            self.centroids = self._initialize_centroids_kmeans_plus_plus(X)
        else:
            self.centroids = self._initialize_centroids_random(X)
        
        # Iterative optimization
        for iteration in range(self.max_iters):
            # Assignment step
            self.labels = self._assign_clusters(X, self.centroids)
            
            # Update step
            new_centroids = self._update_centroids(X, self.labels)
            
            # Check convergence
            centroid_shift = np.sqrt(np.sum((new_centroids - self.centroids) ** 2))
            
            self.centroids = new_centroids
            self.n_iter = iteration + 1
            
            if centroid_shift < self.tol:
                break
        
        # Compute final inertia
        self.inertia = self._compute_inertia(X, self.labels, self.centroids)
        
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for new data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            New data.
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Predicted cluster labels.
        """
        X = np.array(X)
        return self._assign_clusters(X, self.centroids)
    
    def fit_predict(self, X, init='random'):
        """
        Fit model and predict cluster labels.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        init : str, default='random'
            Initialization method: 'random' or 'kmeans++'
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster labels.
        """
        self.fit(X, init=init)
        return self.labels
