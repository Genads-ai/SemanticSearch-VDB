import numpy as np
import os
import sys
from .indexing_strategy import IndexingStrategy
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import memory_profiler
import timeit
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities import compute_recall_at_k
from Quantizers.product_quantizer import ProductQuantizer 
import heapq
import pickle
from joblib import Parallel, delayed

class IVFIndex(IndexingStrategy):
    def __init__(self, vectors, nlist, dimension=70):
        """
        Setting up the most epic index ever.
        Args:
            nlist (int): Number of clusters (Voronoi cells).
            dimension (int): Dimension of the vectors.
            m (int): Number of subspaces.
            nbits (int): Number of bits per subvector (determines the number of centroids
        """
        self.nlist = nlist
        self.dimension = dimension
        self.centroids = None
        self.vectors = vectors
        self.index_inverted_lists = {i: [] for i in range(nlist)}

    def train(self):
        """
        Make those centroids earn their living, go make a cup of tea, this will take a while.
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
        Throw those vectors into their assigned clusters and teach them to stay there.
        Args:
            vectors (np.ndarray): Dataset of shape (num_vectors, dimension).
        """
        print("Assigning vectors to clusters...")
        assignments = np.argmin(cdist(self.vectors, self.centroids, metric="cosine"), axis=1)

        print("Creating inverted lists...")
        # Assign PQ codes and indices to clusters
        for i, cluster_id in enumerate(assignments):
            self.index_inverted_lists[cluster_id].append(i)

        print("Assignment complete!")

    # @memory_profiler.profile
    def search(self, db, query,k=5, nprobe=1, batch_size=100000, n_jobs=-1):
        """
        Here comes the cool part, searching for the nearest neighbors of the query vectors.
        Args:
            query (np.ndarray): Query matrix of shape (num_queries, dimension), where each row is a query vector.
            k (int): Number of nearest neighbors to return for each query vector.
            nprobe (int): Number of clusters to search in.
            batch_size (int): Number of candidates to process in each decoding batch.
        Returns:
            tuple: Two ndarrays:
                - distances (np.ndarray): Shape (num_queries, k), containing distances of the k nearest neighbors.
                - indices (np.ndarray): Shape (num_queries, k), containing indices of the k nearest neighbors in the original vector list.
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Compute distances to centroids and select the top nprobe clusters for each query
        cluster_distances = cdist(query, self.centroids, metric="cosine")
        closest_clusters = np.argsort(cluster_distances, axis=1)[:, :nprobe]
        def process_query(q_idx, query_vector):
            clusters = closest_clusters[q_idx]
            cluster_indices = [self.index_inverted_lists[cluster] for cluster in clusters]
            cluster_indices = np.concatenate(cluster_indices)
            unique_indices = np.unique(cluster_indices)


            # Fetch vectors in batches
            num_candidates = len(unique_indices)
            all_distances = []
            all_indices = []

            for start in range(0, num_candidates, batch_size):
                end = min(start + batch_size, num_candidates)
                batch_indices = unique_indices[start:end]

                # Batch fetch vectors
                batch_vectors = db.get_batch_rows(batch_indices)
                batch_distances = cdist(query_vector, batch_vectors, metric="cosine").squeeze()

                # Keep track of distances and indices
                all_distances.extend(batch_distances)
                all_indices.extend(batch_indices)

            all_distances = np.array(all_distances)
            all_indices = np.array(all_indices)

            # Find the top-k smallest distances
            top_k_indices = np.argsort(all_distances)[:k]
            top_k_distances = all_distances[top_k_indices]
            top_k_original_indices = all_indices[top_k_indices]

            return top_k_distances, top_k_original_indices

        # Parallelize over queries
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_query)(q_idx, query[q_idx].reshape(1, -1))
            for q_idx in range(query.shape[0])
        )

        # Combine results
        all_distances, all_indices = zip(*results)
        return np.array(all_distances), np.array(all_indices)


    def build_index(self):
        nq = self.vectors.shape[0]
        if os.path.exists(f"DBIndexes/ivf_index_{nq}"):
            self.load_index(f"DBIndexes/ivf_index_{nq}")
            return self

        self.train()
        self.add()
        self.save_index(f"DBIndexes/ivf_index_{nq}")
        return self
            
    def save_index(self, path: str):
        """
        Save this masterpiece to a single file so we don’t have to keep track of multiple files.
        Args:
            path (str): Where to save the genius work.
        """
        print(f"Saving index to {path}")
        data = {
            "centroids": self.centroids,
            "index_inverted_lists": self.index_inverted_lists,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print("Index saved successfully!")

    def load_index(self, path: str):
        """
        Lost your index? No worries, let’s load it back and pretend nothing happened.
        Args:
            path (str): Where you last left your precious index.
        """
        print(f"Loading index from {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.centroids = data["centroids"]
        self.index_inverted_lists = data["index_inverted_lists"]
        print("Index loaded successfully!")
        return self