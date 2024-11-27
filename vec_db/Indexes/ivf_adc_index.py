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

class IVFADCIndex(IndexingStrategy):
    def __init__(self, db, nlist, dimension=70, m=35, nbits=8):
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
        self.db = db
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
        kmeans.fit(self.db.get_all_rows())
        self.centroids = kmeans.cluster_centers_
        print("Training complete!")

        # Train the custom PQ quantizer
        print("Training the custom PQ quantizer...")
        self.pq.train(self.db.get_all_rows())
        print("Training custom PQ quantizer complete!")

    def add(self):
        """
        Throw those vectors into their assigned clusters and teach them to stay there.
        Args:
            vectors (np.ndarray): Dataset of shape (num_vectors, dimension).
        """
        print("Assigning vectors to clusters...")
        assignments = np.argmin(cdist(self.db.get_all_rows(), self.centroids, metric="cosine"), axis=1)
        print("Encoding all vectors with PQ...")
        pq_codes = self.pq.encode(self.db.get_all_rows()).astype(np.uint8)

        # Assign PQ codes and indices to clusters
        for i, cluster_id in enumerate(assignments):
            self.index_inverted_lists[cluster_id].append(i)
            self.pq_inverted_lists[cluster_id] = np.concatenate(
                (self.pq_inverted_lists[cluster_id], pq_codes[i:i+1])
            )

        print("Assignment complete!")

    # @memory_profiler.profile
    def search(self, query, db, k=5, nprobe=1, batch_size=1000):
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


        # Here we calculate the distance between the query vectors and the centroids
        # Note that query is a matrix of shape (num_queries, dimension) => it allows us to handle multiple queries at once
        cluster_distances = cdist(query, self.centroids,metric="cosine")  # Shape: (num_queries, nlist)

        # Sort the clusters by distance and select the top nprobe clusters for each query
        closest_clusters = np.argsort(cluster_distances, axis=1)[:, :nprobe]  # Shape: (num_queries, nprobe)

        # Let's get into the action

        all_distances = []
        all_indices = []

        # Here we process each query vector one by one, if it's a single query, no need for this loop but we support multiple queries [OOOOH]
        for q_idx, clusters in enumerate(closest_clusters):
            # This is our query vector for this iteration, we reshape it to be a 2D array of dimensions (1, dimension) to be able to use it in cdist
            query_vector = query[q_idx].reshape(1, -1) 

            # Temporary files for storing encoded candidates and indices
            # delete flag is set to False to prevent deletion of the files after closing them,[ We made them for a reason bakka! :3]
            with tempfile.NamedTemporaryFile(delete=False) as temp_code_file, tempfile.NamedTemporaryFile(delete=False) as temp_index_file:
                code_file_path = temp_code_file.name
                index_file_path = temp_index_file.name

            # Write candidates from all clusters to disk
            with open(code_file_path, "wb") as code_file, open(index_file_path, "wb") as index_file:
                for cluster_id in clusters:
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

            all_distances.append(list(distances))
            all_indices.append(list(indices))

            # Don't wanna leave any traces behind, do we ? ;)
            os.remove(code_file_path)
            os.remove(index_file_path)

        return np.array(all_distances), np.array(all_indices)


    def build_index(self):
        if os.path.exists(f"DBIndexes/ivf_adc_index_{self.db.get_all_rows().shape[0]}_centroids.npy") and \
           os.path.exists(f"DBIndexes/ivf_adc_index_{self.db.get_all_rows().shape[0]}_index_inverted_lists.npy") and \
           os.path.exists(f"DBIndexes/ivf_adc_index_{self.db.get_all_rows().shape[0]}_pq_inverted_lists.npy"):
            self.load_index(f"DBIndexes/ivf_adc_index_{self.db.get_all_rows().shape[0]}")
            return self

        nq, d = self.db.get_all_rows().shape
        self.train()
        self.add()
        self.save_index(f"DBIndexes/ivf_adc_index_{nq}")
        return self

    def save_index(self, path: str):
        """
        Save this masterpiece to a file so we don’t have to redo it all over again.
        Args:
            path (str): Where to save the genius work.
        """
        print(f"Saving index to {path}")
        np.save(f"{path}_centroids.npy", self.centroids)
        np.save(f"{path}_index_inverted_lists.npy", self.index_inverted_lists)
        np.save(f"{path}_pq_inverted_lists.npy", self.pq_inverted_lists)
        self.pq.save(f"{path}_pq_quantizer.pkl")

    def load_index(self, path: str):
        """
        Lost your index? No worries, let’s load it back and pretend nothing happened.
        Args:
            path (str): Where you last left your precious index.
        """
        self.centroids = np.load(f"{path}_centroids.npy")
        self.index_inverted_lists = np.load(f"{path}_index_inverted_lists.npy", allow_pickle=True).item()
        self.pq_inverted_lists = np.load(f"{path}_pq_inverted_lists.npy", allow_pickle=True).item()
        self.pq = self.pq.load(f"{path}_pq_quantizer.pkl")
        return self
