import numpy as np
import os
from .indexing_strategy import IndexingStrategy

class IVFADCIndex(IndexingStrategy):
    def __init__(self):
        self.index = None 

    def build_index(self, vectors: np.ndarray):
        print("Building the IVFADC index")
        
        # Create a folder called DB_Indexes to save the index , this folder should be in the root of the project
        os.makedirs("DBIndexes", exist_ok=True)

        
        # Placeholder for actual IVFADC index building logic
        self.index = self._generate_ivfadc_index(vectors)
        print("IVFADC index built successfully.")

    def save_index(self, path: str):
        if self.index is None:
            raise ValueError("Index not built. Build the index before saving.")
        
        print("Saving the IVFADC index")
        
        # Create a folder called DB_Indexes to save the index , this folder should be in the root of the project
        os.makedirs("DBIndexes", exist_ok=True)
        
        # Placeholder for saving logic
        np.save(path, self.index)
        print(f"IVFADC index saved at: {path}")

    def _generate_ivfadc_index(self, vectors: np.ndarray):
        # Placeholder for actual IVFADC index building logic
        return np.random.random((10, 70))  
    


