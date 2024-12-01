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
from concurrent.futures import ThreadPoolExecutor

class IMIIndex(IndexingStrategy):
    def __init__(self, vectors, nlist, dimension=70):
        assert dimension % 2 == 0, "Dimension must be divisible by 2 for IMI."
        self.nlist = nlist
        self.dimension = dimension
        self.subspace_dim = dimension // 2
        self.vectors = vectors
        self.centroids1 = None
        self.centroids2 = None
        self.index_inverted_lists = {}

    def train(self):
        print("Training IMI centroids...")

        # Split vectors into two subspaces
        subspace1 = self.vectors[:, :self.subspace_dim]
        subspace2 = self.vectors[:, self.subspace_dim:]

        # Train KMeans on each subspace
        kmeans1 = KMeans(n_clusters=self.nlist, random_state=42)
        kmeans2 = KMeans(n_clusters=self.nlist, random_state=42)
        kmeans1.fit(subspace1)
        kmeans2.fit(subspace2)

        self.centroids1 = kmeans1.cluster_centers_
        self.centroids2 = kmeans2.cluster_centers_

        # Initialize the inverted lists for the multi-index
        for i in range(self.nlist):
            for j in range(self.nlist):
                self.index_inverted_lists[(i, j)] = []

        print("Training complete!")

    def add(self):
        print("Assigning vectors to clusters...")

        # Split vectors into two subspaces
        subspace1 = self.vectors[:, :self.subspace_dim]
        subspace2 = self.vectors[:, self.subspace_dim:]

        # Compute assignments for both subspaces
        assignments1 = np.argmin(cdist(subspace1, self.centroids1, metric="cosine"), axis=1)
        assignments2 = np.argmin(cdist(subspace2, self.centroids2, metric="cosine"), axis=1)

        # Create multi-index assignments
        for i, (a1, a2) in enumerate(zip(assignments1, assignments2)):
            self.index_inverted_lists[(a1, a2)].append(i)

        print("Assignment complete!")

    def search(self, db, query, k=5, nprobe=1, batch_size=20000, n_jobs=-1):
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Split query into two subspaces
        subspace1 = query[:, :self.subspace_dim]
        subspace2 = query[:, self.subspace_dim:]

        # Find closest centroids in both subspaces
        cluster_distances1 = cdist(subspace1, self.centroids1, metric="cosine")
        cluster_distances2 = cdist(subspace2, self.centroids2, metric="cosine")
        closest_clusters1 = np.argsort(cluster_distances1, axis=1)[:, :nprobe].flatten()
        closest_clusters2 = np.argsort(cluster_distances2, axis=1)[:, :nprobe].flatten()

        # Combine cluster pairs
        cluster_pairs = [(c1, c2) for c1 in closest_clusters1 for c2 in closest_clusters2]

        # Get indices for all cluster pairs
        cluster_indices = []
        for c1, c2 in cluster_pairs:
            cluster_indices.extend(self.index_inverted_lists[(c1, c2)])
        unique_indices = np.unique(cluster_indices)

        # Function to process a batch
        def process_batch(start, end):
            batch_indices = unique_indices[start:end]
            batch_indices = np.sort(batch_indices)
            batch_vectors = db.get_batch_rows(batch_indices)
            batch_distances = cdist(query, batch_vectors, metric="cosine").squeeze()
            return batch_distances, batch_indices

        # Parallel batch processing
        num_candidates = len(unique_indices)
        batch_ranges = [(start, min(start + batch_size, num_candidates))
                        for start in range(0, num_candidates, batch_size)]

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_batch)(start, end) for start, end in batch_ranges
        )

        all_distances = np.concatenate([res[0] for res in results])
        all_indices = np.concatenate([res[1] for res in results])

        # Find the top-k smallest distances
        top_k_indices = np.argsort(all_distances)[:k]
        top_k_distances = all_distances[top_k_indices]
        top_k_original_indices = all_indices[top_k_indices]

        return np.array(top_k_distances), np.array(top_k_original_indices)

    def build_index(self):
        nq = self.vectors.shape[0]
        if os.path.exists(f"DBIndexes/imi_index_{nq}"):
            self.load_index(f"DBIndexes/imi_index_{nq}")
            return self

        self.train()
        self.add()
        self.save_index(f"DBIndexes/imi_index_{nq}")
        return self

    def save_index(self, path: str):
        print(f"Saving index to {path}")
        data = {
            "centroids1": self.centroids1,
            "centroids2": self.centroids2,
            "index_inverted_lists": self.index_inverted_lists,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print("Index saved successfully!")

    def load_index(self, path: str):
        print(f"Loading index from {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.centroids1 = data["centroids1"]
        self.centroids2 = data["centroids2"]
        self.index_inverted_lists = data["index_inverted_lists"]
        print("Index loaded successfully!")
        return self
