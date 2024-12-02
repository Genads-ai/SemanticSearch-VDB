import numpy as np
import os
import sys
from .indexing_strategy import IndexingStrategy
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import memory_profiler
import timeit
import tempfile
import itertools

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
        num_vectors = self.vectors.shape[0]
        batch_size = 500_000
        subspace1 = self.vectors[:, :self.subspace_dim]
        subspace2 = self.vectors[:, self.subspace_dim:]

        for start_idx in range(0, num_vectors, batch_size):
            end_idx = min(start_idx + batch_size, num_vectors)

            # Process current batch
            batch_subspace1 = subspace1[start_idx:end_idx]
            batch_subspace2 = subspace2[start_idx:end_idx]

            # Compute assignments for the batch
            assignments1 = np.argmin(cdist(batch_subspace1, self.centroids1, metric="cosine"), axis=1)
            assignments2 = np.argmin(cdist(batch_subspace2, self.centroids2, metric="cosine"), axis=1)

            # Create multi-index assignments for the batch
            for i, (a1, a2) in enumerate(zip(assignments1, assignments2), start=start_idx):
                self.index_inverted_lists[(a1, a2)].append(i)


        print("Assignment complete!")

    def search(self, db, query_vector, top_k=5, nprobe=1, batch_size=5000, num_threads=12):
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Split query vector into two subspaces
        query_subspace1 = query_vector[:, :self.subspace_dim]
        query_subspace2 = query_vector[:, self.subspace_dim:]

        # Find closest centroids in both subspaces
        subspace1_distances = cdist(query_subspace1, self.centroids1, metric="cosine")
        subspace2_distances = cdist(query_subspace2, self.centroids2, metric="cosine")
        closest_clusters1 = np.argsort(subspace1_distances, axis=1)[:, :nprobe].flatten()
        closest_clusters2 = np.argsort(subspace2_distances, axis=1)[:, :nprobe].flatten()

        # Generate all possible centroid combinations to look through
        cluster_pairs = np.array(
            [(c1, c2) for c1 in closest_clusters1 for c2 in closest_clusters2]
        )

        # Gather all vectors associated with the generated centroid pairs
        candidate_vectors = np.concatenate(
            [self.index_inverted_lists[tuple(pair)] for pair in cluster_pairs]
        )

        # Sorting for the sliding window
        candidate_vectors.sort()

        # Divide candidate vectors into chunks
        total_vectors = len(candidate_vectors)
        chunk_size = (total_vectors + num_threads - 1) // num_threads
        chunks = [
            candidate_vectors[i * chunk_size : (i + 1) * chunk_size]
            for i in range(num_threads)
        ]

        # Function for processing a single chunk
        def process_chunk(chunk):
            local_heap = []
            left_pointer = 0

            while left_pointer < len(chunk):
                right_pointer = np.searchsorted(
                    chunk, chunk[left_pointer] + batch_size, side="left"
                ) - 1
                block_data = db.get_sequential_block(
                    chunk[left_pointer], chunk[right_pointer] + 1
                )

                # Filter relevant vectors within the block
                relevant_indices = chunk[left_pointer:right_pointer + 1] - chunk[left_pointer]
                block_data = block_data[relevant_indices]

                distances = cdist(query_vector, block_data, metric="cosine").flatten()

                # Select top-k distances within the current window
                if len(distances) > top_k:
                    top_indices = np.argpartition(distances, top_k)[:top_k]
                else:
                    top_indices = np.argsort(distances)

                top_distances = distances[top_indices]

                for dist, idx in zip(top_distances, top_indices):
                    if len(local_heap) < top_k:
                        heapq.heappush(local_heap, (-dist, chunk[left_pointer + idx]))
                    else:
                        heapq.heappushpop(local_heap, (-dist, chunk[left_pointer + idx]))

                left_pointer = right_pointer + 1

            return local_heap

        # Use ThreadPoolExecutor to process chunks in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]

        # Merge results from all threads
        final_heap = []
        for future in futures:
            local_heap = future.result()
            for item in local_heap:
                if len(final_heap) < top_k:
                    heapq.heappush(final_heap, item)
                else:
                    heapq.heappushpop(final_heap, item)

        # Extract top-k results
        final_heap.sort(reverse=True)
        top_k_distances = np.array([-item[0] for item in final_heap[:top_k]])
        top_k_indices = np.array([item[1] for item in final_heap[:top_k]])

        return top_k_distances, top_k_indices
    
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
