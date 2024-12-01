import numpy as np
from sklearn.cluster import KMeans
import pickle
from scipy.spatial.distance import cdist
import faiss
from joblib import Parallel, delayed

class ProductQuantizer:
    def __init__(self, dimension, m, nbits):
        """
        Initialize the Product Quantizer.
        Args:
            dimension (int): Dimension of the input vectors.
            m (int): Number of subspaces.
            nbits (int): Number of bits per subvector (determines the number of centroids).
        """
        assert dimension % m == 0, "Dimension must be divisible by the number of subspaces (m)."
        self.dimension = dimension
        self.m = m
        self.subvector_dim = dimension // m
        self.k = 2**nbits 
        self.centroids = None

    def train(self, data):
        """
        Train the PQ quantizer by clustering each subspace independently.
        Args:
            data (np.ndarray): Input data of shape (num_vectors, dimension).
        """
        num_vectors, dimension = data.shape
        assert dimension == self.dimension, "Data dimension does not match PQ dimension."

        self.centroids = []
        # for i in range(self.m):
        #     subspace_data = data[:, i * self.subvector_dim : (i + 1) * self.subvector_dim]
        #     kmeans = KMeans(n_clusters=self.k, random_state=42)
        #     kmeans.fit(subspace_data)
        #     self.centroids.append(kmeans.cluster_centers_)

        for i in range(self.m):
            # Extract subspace data
            subspace_data = data[:, i * self.subvector_dim: (i + 1) * self.subvector_dim]
            
            # Initialize FAISS KMeans for the subspace
            kmeans = KMeans(n_clusters=self.k, random_state=42)
            kmeans.fit(subspace_data)
            self.centroids.append(kmeans.cluster_centers_)
            

    def encode(self, data):
        """
        Encode the data into PQ codes.
        Args:
            data (np.ndarray): Input data of shape (num_vectors, dimension).
        Returns:
            np.ndarray: PQ codes of shape (num_vectors, m) with dtype=np.uint8.
        """
        def compute_distances_and_codes(subspace_batch, centroids, start_index, i):
            print(f"Processing batch starting at index {start_index} for subspace {i}...")
            distances = cdist(subspace_batch, centroids, metric='euclidean')
            codes = np.argmin(distances, axis=1)
            return range(start_index, start_index + len(codes)), codes

        num_vectors, dimension = data.shape
        assert dimension == self.dimension, "Data dimension does not match PQ dimension."

        codes = np.empty((num_vectors, self.m), dtype=np.uint8)
        for i in range(self.m):
            subspace_data = data[:, i * self.subvector_dim : (i + 1) * self.subvector_dim]
            centroids = self.centroids[i]
            
            # Compute pairwise distances between each subvector and centroids
            # The following is the paralleization of this line using joblib
            # distances = cdist(subspace_data, self.centroids[i], metric='euclidean')

            # Split subspace data into smaller chunks for parallel processing
            batch_size = 100_000
            num_vectors = subspace_data.shape[0]
            batch_indices = [(j, min(j + batch_size, num_vectors)) for j in range(0, num_vectors, batch_size)]
                
            # Parallel computation of distances
            results = Parallel(n_jobs=-1)(
            delayed(compute_distances_and_codes)(subspace_data[start:end], centroids, start, i)
                for start, end in batch_indices
            )
            
            for indices, batch_codes in results:
                codes[indices, i] = batch_codes

        return codes
    
    def decode(self, codes):
        """
        Decode PQ codes back into approximate vectors.
        Args:
            codes (np.ndarray): PQ codes of shape (num_vectors, m).
        Returns:
            np.ndarray: Approximate vectors of shape (num_vectors, dimension).
        """
        num_vectors, m = codes.shape
        assert m == self.m, "Number of subspaces in codes does not match PQ configuration."

        decoded_vectors = np.empty((num_vectors, self.dimension), dtype=np.float32)
        for i in range(self.m):
            subspace_centroids = self.centroids[i][codes[:, i]]
            decoded_vectors[:, i * self.subvector_dim : (i + 1) * self.subvector_dim] = subspace_centroids

        return decoded_vectors

    def save(self, path):
        """
        Save the PQ centroids and configuration to a file.
        Args:
            path (str): File path to save the PQ data.
        """
        data = {
            "dimension": self.dimension,
            "m": self.m,
            "subvector_dim": self.subvector_dim,
            "k": self.k,
            "centroids": self.centroids,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"PQ data saved to {path}.")

    
    def load(self, path):
        """
        Load the PQ centroids and configuration from a file.
        Args:
            path (str): File path to load the PQ data.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.dimension = data["dimension"]
        self.m = data["m"]
        self.subvector_dim = data["subvector_dim"]
        self.k = data["k"]
        self.centroids = data["centroids"]
        print(f"PQ data loaded from {path}.")

        return self
