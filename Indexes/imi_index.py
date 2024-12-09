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
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    def search(self, db, query_vector, top_k=5, nprobe=1, max_difference=10000, batch_limit=2000, pruning_factor=2200):
        def batch_numbers(numbers, max_difference, batch_limit):
            start_index = 0
            batch_count = 0
            n = len(numbers)
            while start_index < n and batch_count < batch_limit:
                min_element = numbers[start_index]
                end_index = start_index

                end_index = np.searchsorted(numbers, min_element + max_difference, side='right')
                end_index = min(end_index, n)

                yield numbers[start_index:end_index]
                batch_count += 1
                start_index = end_index

        def process_batch(batch):
            start_index = batch[0]
            end_index = batch[-1]
            block_data = db.get_sequential_block(start_index, end_index + 1)

            relevant_indices = batch - start_index
            block_data = block_data[relevant_indices]
            block_data = block_data.astype(np.float16, copy=False)

            # Compute cosine distances
            distances = cdist(query_vector, block_data, metric="cosine").flatten().astype(np.float16)

            # Check if there are more than top_k elements in the heap
            if len(distances) <= top_k:
                return list(zip(distances, batch))
            top_indices = np.argpartition(distances, top_k)[:top_k]
            top_indices = top_indices[np.argsort(distances[top_indices])]
            return list(zip(distances[top_indices], batch[top_indices]))

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        query_vector = query_vector.astype(np.float16, copy=False)

        # Split query vector into two subspaces
        query_subspace1 = query_vector[:, :self.subspace_dim]
        query_subspace2 = query_vector[:, self.subspace_dim:]

        # Find closest centroids in both subspaces
        subspace1_distances = cdist(query_subspace1, self.centroids1, metric="cosine").astype(np.float16)
        subspace2_distances = cdist(query_subspace2, self.centroids2, metric="cosine").astype(np.float16)
        closest_clusters1 = np.argsort(subspace1_distances, axis=1)[:, :nprobe].flatten()
        closest_clusters2 = np.argsort(subspace2_distances, axis=1)[:, :nprobe].flatten()

        cluster_pairs = np.array([(c1, c2) for c1 in closest_clusters1 for c2 in closest_clusters2], dtype=np.int32)

        # ----------------------------
        # Early Pruning Step
        # ----------------------------
        # Construct representative vectors for each cluster pair
        representative_vectors = []
        for pair in cluster_pairs:
            # Representative vector: concatenation of the two centroids
            rep_vec = np.concatenate([self.centroids1[pair[0]], self.centroids2[pair[1]]]).astype(np.float16)
            representative_vectors.append(rep_vec)
        representative_vectors = np.array(representative_vectors)

        # Compute distances to representative vectors for pruning
        rep_distances = cdist(query_vector, representative_vectors, metric="cosine").flatten().astype(np.float16)

        # Select top few cluster pairs based on representative vector distance
        # pruning_factor controls how many pairs we keep after pruning
        keep_count = min(pruning_factor, len(cluster_pairs)) - 1
        kept_indices = np.argpartition(rep_distances, keep_count)[:keep_count]
        kept_indices = kept_indices[np.argsort(rep_distances[kept_indices])]
        pruned_cluster_pairs = cluster_pairs[kept_indices]
        # ----------------------------

        # Gather candidate vectors from pruned cluster pairs
        candidate_vectors = np.concatenate(
            [self.index_inverted_lists[tuple(pair)] for pair in pruned_cluster_pairs]
        )

        candidate_vectors.sort()

        batch_generator = batch_numbers(candidate_vectors, max_difference, batch_limit)

        all_candidates = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_batch = {executor.submit(process_batch, batch): batch for batch in batch_generator}
            for future in as_completed(future_to_batch):
                batch_top_k = future.result()
                all_candidates.extend(batch_top_k)

        top_k_global = heapq.nsmallest(top_k, all_candidates, key=lambda x: x[0])
        top_k_distances = np.array([]).astype(np.float16)  # Placeholder for future memory and time savings
        top_k_indices = np.array([item[1] for item in top_k_global], dtype=np.int32)

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
        # print(f"Loading index from {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.centroids1 = data["centroids1"]
        self.centroids2 = data["centroids2"]
        self.index_inverted_lists = data["index_inverted_lists"]
        # print("Index loaded successfully!")
        return self
