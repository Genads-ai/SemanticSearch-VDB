import numpy as np
from abc import ABC, abstractmethod

class IndexingStrategy(ABC):
    @abstractmethod
    def build_index(self, vectors: np.ndarray):
        raise NotImplementedError("Subclasses must implement build_index")
    
    @abstractmethod
    def save_index(self, path: str):
        raise NotImplementedError("Subclasses must implement save_index")