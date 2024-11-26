import numpy as np
import os
import sys
from .indexing_strategy import IndexingStrategy
import faiss
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import memory_profiler
import timeit

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities import compute_recall_at_k

class IVFADCIndex(IndexingStrategy):
    def __init__(self, db, nlist, dimension = 70, m=35, nbits=8):
        """
        Initialize the IVF index.
        Args:
            nlist (int): Number of clusters (Voronoi cells).
            dimension (int): Dimension of the vectors.
        """
        self.nlist = nlist
        self.dimension = dimension
        self.centroids = None
        self.db = db
        self.m = m
        self.nbits = nbits
        self.index_inverted_lists = {i: [] for i in range(nlist)}
        self.pq_inverted_lists = {i: np.empty((0, self.m), dtype=np.uint8) for i in range(nlist)}
        self.pq = None

    def train(self):
        """
        Train the IVF index by clustering the dataset.
        Args:
            vectors (np.ndarray): Dataset of shape (num_vectors, dimension).
        """
        print("Training the IVF index using k-means...")
        kmeans = KMeans(n_clusters=self.nlist, random_state=42)
        kmeans.fit(self.db.get_all_rows())
        self.centroids = kmeans.cluster_centers_
        print("Training complete!")

        # Train the PQ quantizer
        print("Training the PQ quantizer...")
        self.pq = faiss.ProductQuantizer(self.dimension, self.m, self.nbits)
        self.pq.train(self.db.get_all_rows())
        print("Training PQ quantizer complete!")

    def add(self):
        """
        Add vectors to the IVF index.
        Args:
            vectors (np.ndarray): Dataset of shape (num_vectors, dimension).
        """
        print("Assigning vectors to clusters...")
        assignments = np.argmin(cdist(self.db.get_all_rows(), self.centroids), axis=1)
        print("Encoding all vectors with PQ...")
        pq_codes = self.pq.compute_codes(self.db.get_all_rows()).astype(np.uint8)
    
        # Assign PQ codes and indices to clusters
        # # Create two inverted lists: one for the PQ codes and one for the indices
        for i, cluster_id in enumerate(assignments):
            self.index_inverted_lists[cluster_id].append(i)
            self.pq_inverted_lists[cluster_id] = np.concatenate(
                (self.pq_inverted_lists[cluster_id], pq_codes[i:i+1])
            )

            


        # for i, (cluster_id, pq_code) in enumerate(zip(assignments, pq_codes)):
        #     self.inverted_lists[cluster_id].append((i, pq_code))


        print("Assignment complete!")

    @memory_profiler.profile
    def search(self, query, db, k=5, nprobe=1):
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

            start_time = timeit.default_timer()
            # Collect candidates from the closest clusters for this query
            # Add to candidates and candidate_indices
            for cluster_id in clusters:
                candidate_indices.extend(self.index_inverted_lists[cluster_id])
                candidates.extend(self.pq_inverted_lists[cluster_id])


            end_time = timeit.default_timer()
            print(f"Time taken to collect candidates from the closest clusters for query {q_idx}: {end_time - start_time:.4f} seconds")
            
            # Retrieve the actual candidate vectors
            candidates = np.array(candidates)
            decoded_candidates = self.pq.decode(candidates)

            # Compute distances between the query vector and candidates
            query_vector = query[q_idx].reshape(1, -1)  # Extract the single query
            distances = cdist(query_vector, decoded_candidates)[0]
            

            # Pair distances with indices and sort by distance
            start_time = timeit.default_timer()
            sorted_neighbors = sorted(zip(distances, candidate_indices))[:k]
            end_time = timeit.default_timer()
            print(f"Time taken to search for the nearest neighbors: {end_time - start_time:.4f} seconds")


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
        if os.path.exists(f"DBIndexes/ivf_adc_index_{self.db.get_all_rows().shape[0]}_centroids.npy") and os.path.exists(f"DBIndexes/ivf_adc_index_{self.db.get_all_rows().shape[0]}_index_inverted_lists.npy") and os.path.exists(f"DBIndexes/ivf_adc_index_{self.db.get_all_rows().shape[0]}_pq_inverted_lists.npy"):
            self.load_index(f"DBIndexes/ivf_adc_index_{self.db.get_all_rows().shape[0]}")
            return self

        nq, d = self.db.get_all_rows().shape
        self.train()
        self.add()

        
        self.save_index(f"DBIndexes/ivf_adc_index_{nq}")

        return self

    def save_index(self, path: str):
        print(f"Saving index to {path}")
        # Save the centroids and inverted lists
        np.save(f"{path}_centroids.npy", self.centroids)
        np.save(f"{path}_index_inverted_lists.npy", self.index_inverted_lists)
        np.save(f"{path}_pq_inverted_lists.npy", self.pq_inverted_lists)
        # faiss.write_index(self.pq, f"{path}_pq.faiss")


    def load_index(self, path: str):
        # Load the centroids and inverted lists
        self.centroids = np.load(f"{path}_centroids.npy")
        self.index_inverted_lists = np.load(f"{path}_index_inverted_lists.npy", allow_pickle=True).item()
        self.pq_inverted_lists = np.load(f"{path}_pq_inverted_lists.npy", allow_pickle=True).item()
        # self.pq = faiss.read_index(f"{path}_pq.faiss")

        return self
    
    @memory_profiler.profile
    def test_memory_usage(self):
        # Get the size of the vectors
        x = 1
