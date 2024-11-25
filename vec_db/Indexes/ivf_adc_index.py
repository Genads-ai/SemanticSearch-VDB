import numpy as np
import os
import sys
from .indexing_strategy import IndexingStrategy
import faiss


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities import compute_recall_at_k

class IVFADCIndex(IndexingStrategy):
    def __init__(self,index = None) -> None:
        self.index = index

    def build_index(self, vectors: np.ndarray):        
        # Get the size of the vectors
        nq, d = vectors.shape
        # Construct the IVF Index
        index_f = faiss.index_factory(d, "IVF4,Flat")
        index_f.train(vectors)
        index_f.add(vectors)

        # # Construct the ground truth index
        # exact_index = faiss.IndexFlatL2(d)
        # exact_index.add(vectors)

        # # Generate some random query vector
        # xq = np.random.random((100, d)).astype(np.float32)

        # # Define the number of nearest neighbors to search for
        # k = 5

        # D, I = index_f.search(xq, k)


        
        # # Perform an exact search to get ground truth

        # D_exact, I_exact = exact_index.search(xq, k)
        # print(f"Nearest neighbors (Ground Truth): {I_exact}")
        # print(f"Distances (Ground Truth): {D_exact}")
        # print(f"Nearest neighbors: {I}")
        # print(f"Distances: {D}")

        # # Evaluate recall
        # recall = compute_recall_at_k(I_exact, I, k)
        # print(f"Recall@{k}: {recall:.4f}")



        self.index = index_f
        # Name the file in this format indextype_db_size
        # Create the foler DBIndexes in the root directory
        self.save_index(f"DBIndexes/ivf_adc_index_{nq}")
        return index_f

    def save_index(self, path: str):
        print(f"Saving index to {path}")
        faiss.write_index(self.index, path)

    def load_index(self, path: str):
        self.index = faiss.read_index(path)
        return self.index
    
    


