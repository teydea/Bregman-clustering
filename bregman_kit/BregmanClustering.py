import numpy as np

class BregmanDivergence:
    def __init__(self, divergence):
        self.divergence = divergence
        
    def __call__(self, X, Y):
        """
        X, Y -- sets of d-dimensional vectors
        """
        
        if (len(X.shape) < 2):
            X = X.reshape(1, -1)
        if (len(Y.shape) < 2):
            Y = Y.reshape(1, -1)
    
        N_1 = X.shape[0]
        N_2 = Y.shape[0]
        M = np.zeros((N_1, N_2))
        
        for i in range(N_1):
            for j in range(N_2):
                
                M[i, j] = self.divergence(X[i], Y[j]).flatten()
            
        if (M.shape[0] == 1):
            M = M.flatten()
                
        return M

class BregmanHardClustering():
    def __init__(self, n_clusters, divergence, tolerance=1e-7, max_iters=1000, random_state=None):
        self.n_clusters = n_clusters
        self.divergence = BregmanDivergence(divergence)
        self.tolerance = tolerance
        self.max_iters = max_iters

        if (random_state):
            np.random.seed(random_state)
        
        self.centroids = None
        self.weights = None
        
    def fit(self, X, v = None):
        n = X.shape[0]
        d = X.shape[1]
        
        if (v == None):
            v = np.ones(n) / n
            
        self.weights = v
        self.centroids = X[np.random.choice(range(n), size=self.n_clusters, replace=False, p=self.weights)]
        prev_result = 0
        
        for _ in range(self.max_iters):
            X_partition = self.assignment_step(X)    
            self.reestimation_step(X, X_partition)
            
            result = 0
            for i in range(self.n_clusters):
                result += np.dot(self.weights[X_partition[i]], self.divergence(X[X_partition[i]], self.centroids[i]))

            if (abs(prev_result - result) < self.tolerance):
                return
                
            prev_result = result
        
    def assignment_step(self, X):
        X_partition = [[] for _ in range(self.n_clusters)]
        H = self.divergence(X, self.centroids)
        
        for i in range(len(X)):
            h = np.argmin(H[i])
            X_partition[h].append(i)
            
        return X_partition
        
    def reestimation_step(self, X, X_partition):
        for i in range(self.n_clusters):
            if (not X_partition[i]):
                continue
            p_i = np.sum(self.weights[X_partition[i]])
            self.centroids[i] = np.dot(self.weights[X_partition[i]], X[X_partition[i]]) / p_i
        
    def predict(self, X):
        n = X.shape[0]
        labels = np.zeros(n)
        for i in range(n):
            cluster = np.argmin([self.divergence(X[i], c) for c in self.centroids])
            labels[i] = cluster
        return labels.astype(int)


class BregmanSoftClustering():
    def __init__(self, n_clusters, divergence, tolerance=1e-7, max_iters=1000, random_state=None):
        self.n_clusters = n_clusters
        self.divergence = BregmanDivergence(divergence)
        self.tolerance = tolerance
        self.max_iters = max_iters

        if (random_state):
            np.random.seed(random_state)
        
        self.centroids = None
        self.weights = None
        
    def fit(self, X):
        n = X.shape[0]
        d = X.shape[1]
        
        self.weights = abs(np.random.random(self.n_clusters))
        self.weights /= np.sum(self.weights)
        self.centroids = X[np.random.choice(range(n), size=self.n_clusters, replace=False)]
        
        prev_result = 0
        
        for _ in range(self.max_iters):
            
            P = self.expectation_step(X)
            self.maximixation_step(X, P)
            
            result = 0
            
            for i in range(n):
                result += np.dot(self.weights, np.exp(-self.divergence(X[i], self.centroids)))
                
            if (abs(prev_result - result) < self.tolerance):
                return
        
            prev_result = result

        
    def expectation_step(self, X):
        n = X.shape[0]
        P = np.zeros((n, self.n_clusters))
        
        for i in range(n):
            denominator = np.dot(self.weights, np.exp(-self.divergence(X[i], self.centroids)))
            for j in range(self.n_clusters):
                P[i, j] = (self.weights[j] * np.exp(-self.divergence(X[i], self.centroids[j]))) / denominator
           
        return P
    
    def maximixation_step(self, X, P):
        for j in range(self.n_clusters):
            self.weights[j] = np.mean(P[:, j])
            self.centroids[j] = np.dot(P[:, j], X) / np.sum(P[:, j])
            
    def predict_probs(self, X):
        return self.expectation_step(X)
    
    def predict(self, X):
        probs = self.expectation_step(X)
        return np.argmax(probs, axis=1)
