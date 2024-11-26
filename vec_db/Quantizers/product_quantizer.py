import numpy as np
from sklearn.cluster import KMeans
import pickle
from scipy.spatial.distance import cdist

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
        for i in range(self.m):
            subspace_data = data[:, i * self.subvector_dim : (i + 1) * self.subvector_dim]
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
        num_vectors, dimension = data.shape
        assert dimension == self.dimension, "Data dimension does not match PQ dimension."

        codes = np.empty((num_vectors, self.m), dtype=np.uint8)
        for i in range(self.m):
            subspace_data = data[:, i * self.subvector_dim : (i + 1) * self.subvector_dim]
            
            # Compute pairwise distances between each subvector and centroids
            distances = cdist(subspace_data, self.centroids[i], metric='euclidean')
            
            # Assign each subvector to the closest centroid
            codes[:, i] = np.argmin(distances, axis=1)

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
