from typing import Dict, List, Annotated
import numpy as np
import os
import Indexes.ivf_adc_index as ivf_adc_index
import Indexes.flat_index as flat_index
from utilities import compute_recall_at_k

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        print("Initializing the VecDB")
        self.db_path = database_file_path
        self.index_path = index_file_path
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        scores = []
        num_records = self._get_num_records()
        # here we assume that the row number is the ID of each vector
        for row_num in range(num_records):
            vector = self.get_one_row(row_num)
            score = self._cal_score(query, vector)
            scores.append((score, row_num))
        # here we assume that if two rows have the same score, return the lowest ID
        scores = sorted(scores, reverse=True)[:top_k]
        return [s[1] for s in scores]
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self,indexing_strategy = ivf_adc_index.IVFADCIndex()) -> None:
        vectors = self.get_all_rows()

        indexing_strategy.build_index(vectors)



if __name__ == "__main__":
    db_size = 10000
    db_file_path = f"Databases"
    db_file_name = f"DB_{db_size}.dat"
    index_file_path = f"DBIndexes"
    # Turn this flag on if you want to create a new database or create a new index file
    is_new_db = True


    # Create a directory in the parent directory of the current working directory for the databases & indexes
    if not os.path.exists(db_file_path):
        os.mkdir(db_file_path)
    if not os.path.exists(index_file_path):
        os.mkdir(index_file_path)

    db_file_path = db_file_path + "/" + db_file_name

    # Create a new database
    vec_db = VecDB(database_file_path=db_file_path, index_file_path=index_file_path, new_db=is_new_db, db_size=db_size)

    ivf_adc_idx = ivf_adc_index.IVFADCIndex().load_index(f"DBIndexes/ivf_adc_index_{db_size}")

    # Build the flat index
    ground_truth_idx = flat_index.FlatIndex().build_index(vec_db.get_all_rows())

    # Perform a search using the IVFADC index
    query = np.random.random((10, DIMENSION)).astype(np.float32)
    k = 5
    D, I = ivf_adc_idx.search(query, k)

    # Perform an exact search to get ground truth
    D_exact, I_exact = ground_truth_idx.search(query, k)

    print(f"Nearest neighbors (Ground Truth): {I_exact}")
    print(f"Distances (Ground Truth): {D_exact}")
    print(f"Nearest neighbors: {I}")
    print(f"Distances: {D}")

    # Evaluate recall
    recall = compute_recall_at_k(I_exact, I, k)
    print(f"Recall@{k}: {recall:.4f}")




