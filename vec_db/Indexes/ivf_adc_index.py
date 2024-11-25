import numpy as np
import os
import sys
from .indexing_strategy import IndexingStrategy
import faiss
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities import compute_recall_at_k

class IVFADCIndex(IndexingStrategy):
    def __init__(self, vectors, nlist, dimension = 70):
        """
        Initialize the IVF index.
        Args:
            nlist (int): Number of clusters (Voronoi cells).
            dimension (int): Dimension of the vectors.
        """
        self.nlist = nlist
        self.dimension = dimension
        self.centroids = None
        self.vectors = vectors
        self.inverted_lists = {i: [] for i in range(nlist)}

    def train(self):
        """
        Train the IVF index by clustering the dataset.
        Args:
            vectors (np.ndarray): Dataset of shape (num_vectors, dimension).
        """
        print("Training the IVF index using k-means...")
        kmeans = KMeans(n_clusters=self.nlist, random_state=42)
        kmeans.fit(self.vectors)
        self.centroids = kmeans.cluster_centers_
        print("Training complete!")

    def add(self):
        """
        Add vectors to the IVF index.
        Args:
            vectors (np.ndarray): Dataset of shape (num_vectors, dimension).
        """
        print("Assigning vectors to clusters...")
        assignments = np.argmin(cdist(self.vectors, self.centroids), axis=1)
        for i, cluster_id in enumerate(assignments):
            self.inverted_lists[cluster_id].append(i)
        print("Assignment complete!")

    def search(self, query, k=5, nprobe=1):
        """
        Search for the k nearest neighbors for each query vector.
        Args:
            query (np.ndarray): Query matrix of shape (num_queries, dimension), where each row is a query vector.
            k (int): Number of nearest neighbors to return for each query vector.
            nprobe (int): Number of clusters to search in.
        Returns:
            tuple: Two ndarrays:
                - distances (np.ndarray): Shape (num_queries, k), containing distances of the k nearest neighbors.
                - indices (np.ndarray): Shape (num_queries, k), containing indices of the k nearest neighbors in the original vector list.
        """

        # Compute distances from all queries to centroids
        cluster_distances = cdist(query, self.centroids)  # Shape: (num_queries, num_clusters)
        closest_clusters = np.argsort(cluster_distances, axis=1)[:, :nprobe]  # Shape: (num_queries, nprobe)

        # Initialize results
        all_distances = []
        all_indices = []

        # Process each query vector
        for q_idx, clusters in enumerate(closest_clusters):
            candidates = []
            candidate_indices = []

            # Collect candidates from the closest clusters for this query
            for cluster_id in clusters:
                candidate_indices.extend(self.inverted_lists[cluster_id])
            
            # Retrieve the actual candidate vectors
            candidates = np.array([self.vectors[i] for i in candidate_indices])

            # Compute distances between the query vector and candidates
            query_vector = query[q_idx].reshape(1, -1)  # Extract the single query
            distances = cdist(query_vector, candidates)[0]

            # Pair distances with indices and sort by distance
            sorted_neighbors = sorted(zip(distances, candidate_indices))[:k]

            # Separate distances and indices
            distances, indices = zip(*sorted_neighbors)
            all_distances.append(list(distances))
            all_indices.append(list(indices))

        # Convert results to ndarrays
        all_distances = np.array(all_distances)
        all_indices = np.array(all_indices)

        return all_distances, all_indices


    def build_index(self):        
        # Get the size of the vectors
        # Check if the index has already been built if it's already been built load it
        if os.path.exists(f"DBIndexes/ivf_adc_index_{self.vectors.shape[0]}_centroids.npy") and os.path.exists(f"DBIndexes/ivf_adc_index_{self.vectors.shape[0]}_inverted_lists.npy"):
            self.load_index(f"DBIndexes/ivf_adc_index_{self.vectors.shape[0]}")
            return self

        nq, d = self.vectors.shape
        self.train()
        self.add()

        
        self.save_index(f"DBIndexes/ivf_adc_index_{nq}")

        return self

    def save_index(self, path: str):
        print(f"Saving index to {path}")
        # Save the centroids and inverted lists
        np.save(f"{path}_centroids.npy", self.centroids)
        np.save(f"{path}_inverted_lists.npy", self.inverted_lists)

    def load_index(self, path: str):
        # Load the centroids and inverted lists
        self.centroids = np.load(f"{path}_centroids.npy")
        self.inverted_lists = np.load(f"{path}_inverted_lists.npy", allow_pickle=True).item()

        return self
    
    


