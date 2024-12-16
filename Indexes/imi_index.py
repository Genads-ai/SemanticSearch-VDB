import numpy as np
import os
import sys
from .indexing_strategy import IndexingStrategy
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import memory_profiler
import timeit
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities import compute_recall_at_k
from Quantizers.product_quantizer import ProductQuantizer 
import heapq
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed


class IMIIndex(IndexingStrategy):
    def __init__(self, vectors, nlist, dimension=70,index_path=None):
        assert dimension % 2 == 0, "Dimension must be divisible by 2 for IMI."
        self.nlist = nlist # The Number Of Centroids in each of the 2 Subspaces
        self.dimension = dimension # The Dimension of the Vectors Const = 70
        self.subspace_dim = dimension // 2 # The Dimension of each of the 2 Subspaces
        self.vectors = vectors # The Vectors to be Indexed Only Used for Training
        # The combination of the centroids of the 2 subspaces gives us 65536 centroids
        self.centroids1 = None # The Centroids of the 1st Subspace Only Used for Training = 256
        self.centroids2 = None # The Centroids of the 2nd Subspace Only Used for Training = 256
        self.index_inverted_lists = {} # The Inverted Lists for the Multi-Index = 65536 Keys Each Key is a Tuple of 2 Centroids (C1,C2) and the Value is a List of Integers which are the Indices of the Vectors that are assigned to this key
        self.index_path = index_path # The Path to the Index File / Practically Unused in the final state
        self.index_offsets = None  # Loading Only The Top Level Index
        inverted_list_dir = os.path.join("DBIndexes", f"imi_index_{self.vectors.shape[0]//10**6}M")
        index_file_path = os.path.join(inverted_list_dir, "index_offsets.bin")
        with open(index_file_path, "rb") as f:
            self.index_offsets = np.fromfile(f, dtype=np.int32).reshape(-1, 2)

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
        # This Basically Assigns Each Vector to a Pair of centroids.
        # So if the first half is close to C4 in the first subspace and the second half is close to C5 in the second subspace then this vector is assigned to the pair (C4,C5) = (4,5)
        print("Assigning vectors to clusters...")
        # Split vectors into two subspaces
        num_vectors = self.vectors.shape[0]
        batch_size = 500_000
        # Divide the 70 dimensional vectors into 2*35 dimensional vectors
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

    def search(self, db, query_vector, top_k=5, nprobe=1, max_difference=7500, batch_limit=3000, pruning_factor=2250):
        # Generator to yield batches and save memory
        def batch_numbers(numbers, max_difference, batch_limit):
            start_index = 0
            batch_count = 0
            n = len(numbers)
            while start_index < n and batch_count < batch_limit:
                min_element = numbers[start_index]
                end_index = start_index

                # Returns the index of the first element greater than or equal to min_element + max_difference
                end_index = np.searchsorted(numbers, min_element + max_difference, side='right')
                end_index = min(end_index, n)

                yield numbers[start_index:end_index]

                batch_count += 1
                start_index = end_index

        # Function to process a batch of indices : 1- Retrieve the vectors from the database 2- Compute the cosine similarity 3- Return the top k
        def process_batch(batch):
            start_index = batch[0]
            end_index = batch[-1]
            block_data = db.get_sequential_block(start_index, end_index + 1)

            relevant_indices = batch - start_index
            block_data = block_data[relevant_indices]
            block_data = block_data.astype(np.float16,copy = False) 

            # Compute cosine distances
            distances = cdist(query_vector, block_data, metric="cosine").flatten().astype(np.float16,copy = False)

            # Check if there are more than top_k elements in the heap
            if len(distances) <= top_k:
                return list(zip(distances, batch))
            
            # We use argpartition for efficiency as there's no need to sort the whole array so we get the top k elements in O(n) time 
            top_indices = np.argpartition(distances, top_k)[:top_k]
            top_indices = top_indices[np.argsort(distances[top_indices])]
            return list(zip(distances[top_indices], batch[top_indices]))


        # This is useless but this function initially supported getting a Nx70 Queries Array and looped through them
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)


        query_vector = query_vector.astype(np.float16,copy = False)

        # Split query vector into two subspaces
        query_subspace1 = query_vector[:, :self.subspace_dim]
        query_subspace2 = query_vector[:, self.subspace_dim:]

        # Reads The Centroids From The Disk
        centroids = self.load_centroids()
        centroids1 = centroids["centroids1"]
        centroids2 = centroids["centroids2"]

        # Find closest centroids in both subspaces
        subspace1_distances = cdist(query_subspace1, centroids1, metric="cosine").flatten()
        subspace2_distances = cdist(query_subspace2, centroids2, metric="cosine").flatten()

        # Create a grid of combined distances using broadcasting to avoid memory redundancy
        combined_distances = subspace1_distances[:, None] + subspace2_distances[None, :]

        # Flatten combined distances and generate centroid pair indices
        combined_distances_flat = combined_distances.ravel()
        centroid_pairs = np.indices((256, 256)).reshape(2, -1).T # This Produces a 2D array of all possible centroid pairs something like this [(0,0), (0,1), ..., (255,255)]

        # Select the top nprobe * nprobe centroid pairs
        top_indices = np.argpartition(combined_distances_flat, nprobe * nprobe)[:nprobe * nprobe]
        top_indices = top_indices[np.argsort(combined_distances_flat[top_indices])]

        # Use the top indices to extract cluster pairs
        cluster_pairs = centroid_pairs[top_indices]

        # ----------------------------  
        # Early Pruning Step
        # ----------------------------
        # Construct representative vectors for each cluster pair
        representative_vectors = np.empty((len(cluster_pairs), self.dimension), dtype=np.float16)
        for idx, pair in enumerate(cluster_pairs):
            representative_vectors[idx] = np.concatenate([
                centroids1[pair[0]], 
                centroids2[pair[1]]
            ])

        # Compute distances to representative vectors for pruning
        rep_distances = cdist(query_vector, representative_vectors, metric="cosine").flatten()

        # Select top few cluster pairs based on representative vector distance
        # pruning_factor controls how many pairs we keep after pruning
        keep_count = min(pruning_factor, len(cluster_pairs)) - 1
        kept_indices = np.argpartition(rep_distances, keep_count)[:keep_count]
        kept_indices = kept_indices[np.argsort(rep_distances[kept_indices])]  
        pruned_cluster_pairs = cluster_pairs[kept_indices]
        # ----------------------------

        candidate_vectors = self.load_index_inverted_lists(pruned_cluster_pairs)
            
        candidate_vectors.sort()


        batch_generator = batch_numbers(candidate_vectors, max_difference, batch_limit)

        all_candidates = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_batch = {executor.submit(process_batch, batch): batch for batch in batch_generator}
            for future in as_completed(future_to_batch):
                batch_top_k = future.result()
                all_candidates.extend(batch_top_k)

        top_k_global = heapq.nsmallest(top_k, all_candidates, key=lambda x: x[0])
        top_k_distances = np.array([item[0] for item in top_k_global]) # [-item[0] for item in local_heap] This should be in here but removed to save memory & time
        top_k_indices = np.array([item[1] for item in top_k_global])

        return top_k_distances, top_k_indices


    def build_index(self):
        # Another Unused Function After Restructuring
        nq = self.vectors.shape[0]
        if os.path.exists(self.index_path):
            # self.load_index(self.index_path)
            return self

        return self

        self.train()
        self.add()
        self.save_index(self.index_path)
        return self

    def save_index(self, path: str):
        # Was Used To Save The Initial State Which Was Restructured Later
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
        # We Stopped Using This It's Not Called Anywhere in the code
        # print(f"Loading index from {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.centroids1 = data["centroids1"]
        self.centroids2 = data["centroids2"]
        self.index_inverted_lists = data["index_inverted_lists"]
        # print("Index loaded successfully!")
        return self

    def restructure_pickle(pickle_path, output_dir, size):
        # This was used to restructure the initial pickle files to handle partial access as it was changed after the initial confirmation
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        centroids_path = os.path.join(output_dir, f"centroids_{size//10**6}M.pkl")
        centroids_data = {"centroids1": data["centroids1"], "centroids2": data["centroids2"]}
        with open(centroids_path, "wb") as f:
            pickle.dump(centroids_data, f)
        print(f"Centroids saved to {centroids_path}")

        inverted_list_dir = os.path.join(output_dir, f"imi_index_{size//10**6}M")
        os.makedirs(inverted_list_dir, exist_ok=True)

        concatenated_values = []
        # We Have 256 * 256 = 65536 Possible Keys For Each We Do Store where it starts and how many elements it has hence the 2 in the shape
        index_offsets = np.zeros((256 * 256, 2), dtype=np.int32)

        for key, value in data["index_inverted_lists"].items():
            start = len(concatenated_values)
            length = len(value)
            concatenated_values.extend(value)
            index = key[0] * 256 + key[1]
            index_offsets[index] = [start, length]

        concatenated_values_path = os.path.join(inverted_list_dir, "concatenated_values.bin")
        index_file_path = os.path.join(inverted_list_dir, "index_offsets.bin")

        with open(concatenated_values_path, "wb") as f:
            np.array(concatenated_values, dtype=np.int32).tofile(f)
        print(f"Concatenated values saved to {concatenated_values_path}")

        with open(index_file_path, "wb") as f:
            index_offsets.tofile(f)
        print(f"Index offsets saved to {index_file_path}")



    def load_centroids(self):
        centroids_path = os.path.join("DBIndexes", f"centroids_{self.vectors.shape[0]//10**6}M.pkl")
        with open(centroids_path, "rb") as f:
            centroids_data = pickle.load(f)
        return {"centroids1": centroids_data["centroids1"], "centroids2": centroids_data["centroids2"]}


    def load_index_inverted_lists(self, keys=None):
        inverted_list_dir = os.path.join("DBIndexes", f"imi_index_{self.vectors.shape[0]//10**6}M")
        concatenated_values_path = os.path.join(inverted_list_dir, "concatenated_values.bin")

        # This can be read here and it won't affect the performance but why don't we abuse the fact that this is the top level index
        # inverted_list_dir = os.path.join("DBIndexes", f"imi_index_{self.vectors.shape[0]//10**6}M")
        # index_file_path = os.path.join(inverted_list_dir, "index_offsets.bin")
        # with open(index_file_path, "rb") as f:
        #     self.index_offsets = np.fromfile(f, dtype=np.int32).reshape(-1, 2)


        # Sort The Keys For Batching Sequential Access
        total_length = 0
        keys = sorted(keys, key=lambda key: tuple(key) if isinstance(key, (list, np.ndarray)) else key)
        for key in keys:
            key = tuple(key) if isinstance(key, (list, np.ndarray)) else key
            index = key[0] * 256 + key[1]
            start, length = self.index_offsets[index]
            total_length += length

        candidate_vectors = np.empty((total_length,), dtype=np.int32)


        # Same as window generation technique used in the search function
        def batch_keys(keys, batch_size=25):
            for i in range(0, len(keys), batch_size):
                yield keys[i:i+batch_size]

        array_start = 0 # Points at the first available index in the candidate_vectors array to be able to fill it with retrieved vectors' indices
        if keys is not None:
            for key_batch in batch_keys(keys):
                batch_starts = []
                batch_lengths = []

                for key in key_batch:
                    key = tuple(key) if isinstance(key, (list, np.ndarray)) else key
                    index = key[0] * 256 + key[1]
                    start, length = self.index_offsets[index]
                    batch_starts.append(start)
                    batch_lengths.append(length)

                min_start = min(batch_starts)
                max_end = max(start + length for start, length in zip(batch_starts, batch_lengths))

                block = np.memmap(concatenated_values_path, dtype=np.int32, mode='r', offset=min_start * 4, shape=(max_end - min_start,))

                for start, length in zip(batch_starts, batch_lengths):
                    candidate_vectors[array_start:array_start+length] = block[start-min_start:start-min_start+length]
                    array_start += length

        return candidate_vectors