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
from concurrent.futures import ThreadPoolExecutor, as_completed

class IVFADCIndex(IndexingStrategy):
    def __init__(self, vectors, nlist, dimension=70, m=35, nbits=8):
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
        self.m = m
        self.nbits = nbits
        self.index_inverted_lists = {i: [] for i in range(nlist)}
        self.pq_inverted_lists = {i: np.empty((0, self.m), dtype=np.uint8) for i in range(nlist)}
        self.pq = ProductQuantizer(self.dimension, self.m, self.nbits) 

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

        # Train the custom PQ quantizer
        print("Training the custom PQ quantizer...")
        self.pq.train(self.vectors)
        print("Training custom PQ quantizer complete!")

    def add(self):
        """
        Throw those vectors into their assigned clusters and teach them to stay there.
        Args:
            vectors (np.ndarray): Dataset of shape (num_vectors, dimension).
        """
        print("Assigning vectors to clusters...")
        assignments = np.argmin(cdist(self.vectors, self.centroids, metric="cosine"), axis=1)
        print("Encoding all vectors with PQ...")
        pq_codes = self.pq.encode(self.vectors).astype(np.uint8)

        # Assign PQ codes and indices to clusters
        for i, cluster_id in enumerate(assignments):
            self.index_inverted_lists[cluster_id].append(i)
            self.pq_inverted_lists[cluster_id] = np.concatenate(
                (self.pq_inverted_lists[cluster_id], pq_codes[i:i+1])
            )

        print("Assignment complete!")

    # @memory_profiler.profile

    def searchFile(self, db, query, k=5, nprobe=1, batch_size=2500):
        """
        Search for the nearest neighbors of the query vector using multithreading.
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
        # If the query is a single vector, reshape it to a 2D array
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Calculate distances between the query vector and centroids
        cluster_distances = cdist(query, self.centroids, metric="cosine")  # Shape: (1, nlist)

        # Select the top nprobe clusters
        closest_clusters = np.argsort(cluster_distances[0])[:nprobe]

        # Min-heap to store the top-k results
        min_heap = []
        
        def process_cluster(cluster_id):
            """Process candidates from a single cluster and return top-k for that cluster."""
            if len(self.index_inverted_lists[cluster_id]) == 0:
                return []

            cluster_codes = np.array(self.pq_inverted_lists[cluster_id], dtype=np.uint8)
            cluster_indices = np.array(self.index_inverted_lists[cluster_id], dtype=np.int32)

            cluster_heap = []

            # Process candidates in batches
            for i in range(0, len(cluster_codes), batch_size):
                pq_batch = cluster_codes[i:i + batch_size]
                index_batch = cluster_indices[i:i + batch_size]

                # Decode the PQ batch
                decoded_batch = self.pq.decode(pq_batch)

                # Compute distances for this batch
                batch_distances = cdist(query, decoded_batch, metric="cosine")[0]

                # Merge into the top-k results using a heap
                for dist, idx in zip(batch_distances, index_batch):
                    if len(cluster_heap) < k:
                        heapq.heappush(cluster_heap, (-dist, idx))
                    else:
                        heapq.heappushpop(cluster_heap, (-dist, idx))

            return cluster_heap

        # Use ThreadPoolExecutor to process clusters in parallel
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_cluster, cluster_id) for cluster_id in closest_clusters]

            for future in as_completed(futures):
                cluster_heap = future.result()
                # Merge results from each cluster into the global heap
                for dist, idx in cluster_heap:
                    if len(min_heap) < k:
                        heapq.heappush(min_heap, (dist, idx))
                    else:
                        heapq.heappushpop(min_heap, (dist, idx))

        # Extract the top-k results from the global heap
        top_k = heapq.nlargest(k, min_heap)
        distances, indices = zip(*[(-d, i) for d, i in top_k])

        return np.array(distances), np.array(indices)
    
    def search(self, db, query, k=5, nprobe=1, batch_size=2500):
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

        # Not to brag, but this is the most optimized search function in the history of search functions
        # The min heap is doing wonders here 8) : Best ideas come just before you go to sleep

        # If the query is a single vector, we need to reshape it to be a 2D array of shape (1, dimension) to be able to use it in cdist
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Here we calculate the distance between the query vectors and the centroids
        # Note that query is a matrix of shape (num_queries, dimension) => it allows us to handle multiple queries at once
        cluster_distances = cdist(query, self.centroids,metric="cosine")  # Shape: (num_queries, nlist)

        # Sort the clusters by distance and select the top nprobe clusters for each query
        closest_clusters = np.argsort(cluster_distances, axis=1)[:, :nprobe].flatten()
        # Let's get into the action


        # This is our query vector for this iteration, we reshape it to be a 2D array of dimensions (1, dimension) to be able to use it in cdist
        query_vector = query.reshape(1, -1) 

        # Temporary files for storing encoded candidates and indices
        # delete flag is set to False to prevent deletion of the files after closing them,[ We made them for a reason bakka! :3]
        with tempfile.NamedTemporaryFile(delete=False) as temp_code_file, tempfile.NamedTemporaryFile(delete=False) as temp_index_file:
            code_file_path = temp_code_file.name
            index_file_path = temp_index_file.name

        # Write candidates from all clusters to disk
        with open(code_file_path, "wb") as code_file, open(index_file_path, "wb") as index_file:
            for cluster_id in closest_clusters:
                # Write quantized vectors codes to disk which were assigned to this cluster
                cluster_codes = np.array(self.pq_inverted_lists[cluster_id], dtype=np.uint8)
                code_file.write(cluster_codes.tobytes())

                # Write indices of the same vectors to disk
                cluster_indices = np.array(self.index_inverted_lists[cluster_id], dtype=np.int32)
                index_file.write(cluster_indices.tobytes())

        # Incrementally decode and compute distances from disk
        # To store the top-k results as (distance, index) pairs
        # The main idea here is to use a heap to store the top-k results, this way we can keep the heap size to k and avoid sorting the whole list
        # We do this due to the RAM constraints, we can't store all the vectors and indices in memory so we keep the top-k only which are the ones we care about <3
        min_heap = []  
        with open(code_file_path, "rb") as code_file, open(index_file_path, "rb") as index_file:
            while True:
                # Read a chunk of PQ codes and indices
                code_chunk = code_file.read(batch_size * self.m)  # PQ codes
                index_chunk = index_file.read(batch_size * 4)  # Indices (int32)

                if not code_chunk or not index_chunk:
                    break

                pq_batch = np.frombuffer(code_chunk, dtype=np.uint8).reshape(-1, self.m)
                index_batch = np.frombuffer(index_chunk, dtype=np.int32)

                # Decode the PQ batch
                decoded_batch = self.pq.decode(pq_batch)

                # Compute distances for this batch
                batch_distances = cdist(query_vector, decoded_batch, metric="cosine")[0]

                # Merge into the top-k results using a heap
                for dist, idx in zip(batch_distances, index_batch):
                    if len(min_heap) < k:
                        heapq.heappush(min_heap, (-dist, idx))
                    else:
                        heapq.heappushpop(min_heap, (-dist, idx))

            # Extract top-k results from the heap
            # You would argue that this is not necessary, and you're right, but it will look cooler this way
            # top_k is already small, so sorting it won't take much time [Stop arguing with me, I'm the one writing the code here >:( ]
            top_k = sorted((-d, i) for d, i in min_heap) 
            distances, indices = zip(*top_k)

        # Don't wanna leave any traces behind, do we ? ;)
        os.remove(code_file_path)
        os.remove(index_file_path)

        return np.array(distances), np.array(indices)


    def build_index(self):
        nq = self.vectors.shape[0]
        if os.path.exists(f"DBIndexes/ivf_adc_index_{nq}"):
            print("Index already exists, loading it...")
            self.load_index(f"DBIndexes/ivf_adc_index_{nq}")
            return self

        self.train()
        self.add()
        self.save_index(f"DBIndexes/ivf_adc_index_{nq}")
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
            "pq_inverted_lists": self.pq_inverted_lists,
            "pq": self.pq,
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
        self.pq_inverted_lists = data["pq_inverted_lists"]
        self.pq = data["pq"]
        print("Index loaded successfully!")
        return self