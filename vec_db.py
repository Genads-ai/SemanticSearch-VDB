from typing import Dict, List, Annotated
import numpy as np
import os
import Indexes.ivf_adc_index as ivf_adc_index
import Indexes.flat_index as flat_index
import Indexes.ivf_index as ivf_index
import Indexes.imi_index as imi_index
from utilities import compute_recall_at_k,measure_memory_usage
import timeit

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70
N_PROBE = 80

class VecDB:
    def __init__(self, database_file_path = "Databases/DB_1000000.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        print("Initializing the VecDB")
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.index = None
        self.file_handle = open(self.db_path, 'r+b') if not new_db else None
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
        else:
            if self.file_handle is None:
                self.file_handle = open(self.db_path, 'r+b')
            if "ivf" in self.index_path:
                self.index = self._build_index(ivf_adc_index.IVFADCIndex(vectors=self.get_all_rows(),nlist=256,dimension=70))
            else:
                self.index = self._build_index(imi_index.IMIIndex(vectors=self.get_all_rows(),nlist=256,dimension=70,index_path=index_file_path)) 

    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        if "ivf" in self.index_path:
            self.index = self._build_index(ivf_adc_index.IVFADCIndex(vectors=self.get_all_rows(),nlist=256,dimension=70))
        else:
            self.index = self._build_index(imi_index.IMIIndex(vectors=self.get_all_rows(),nlist=256,dimension=70,index_path=self.index_path))
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
   

    def get_sequential_block(self, left_index: int, right_index: int) -> np.ndarray:
        try:
            if left_index < 0 or right_index <= left_index:
                raise ValueError("Invalid range: left_index must be >= 0 and right_index must be > left_index.")

            total_rows = self._get_num_records()

            if right_index > total_rows:
                raise ValueError(f"Invalid range: right_index {right_index} exceeds total rows {total_rows}.")

            num_vectors = np.int64(right_index) - np.int64(left_index)
            byte_offset = np.int64(left_index) * np.int64(DIMENSION) * np.int64(ELEMENT_SIZE)
            byte_size = np.int64(num_vectors) * np.int64(DIMENSION) * np.int64(ELEMENT_SIZE)

            # Seek to the start offset
            self.file_handle.seek(byte_offset)

            # Read the required bytes
            bytes_read = self.file_handle.read(byte_size)
            if len(bytes_read) != byte_size:
                raise IOError(f"Failed to read the complete block from {left_index} to {right_index}.")

            # Convert bytes to NumPy array
            block = np.frombuffer(bytes_read, dtype=np.float32).reshape((num_vectors, DIMENSION))

            return block
        except Exception as e:
            print(f"An error occurred while fetching the sequential block: {e}")
            return np.array([])

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    # def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
    #     scores = []
    #     num_records = self._get_num_records()
    #     # here we assume that the row number is the ID of each vector
    #     for row_num in range(num_records):
    #         vector = self.get_one_row(row_num)
    #         score = self._cal_score(query, vector)
    #         scores.append((score, row_num))
    #     # here we assume that if two rows have the same score, return the lowest ID
    #     scores = sorted(scores, reverse=True)[:top_k]
    #     return [s[1] for s in scores]

    # Define the retrieval function which uses the index to retrieve the nearest neighbors
    def retrieve(self, query: np.ndarray, top_k: int) -> List[int]:
        # Search for the nearest neighbors
        D, I = self.index.search(self,query, top_k, nprobe=N_PROBE)
        return I.tolist()
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self,indexing_strategy,nlist=256,dimension=70):
        return indexing_strategy.build_index()



if __name__ == "__main__":
    db_size = 10_000_000
    db_file_path = f"Databases"
    db_file_name = f"DB_{db_size}.dat"
    index_file_path = f"DBIndexes"
    # Turn this flag on if you want to create a new database or create a new index file``
    is_new_db = True


    # Create a directory in the parent directory of the current working directory for the databases & indexes
    if not os.path.exists(db_file_path):
        os.mkdir(db_file_path)
    if not os.path.exists(index_file_path):
        os.mkdir(index_file_path)

    db_file_path = db_file_path + "/" + db_file_name

    # Create a new database
    vec_db = VecDB(database_file_path=db_file_path, index_file_path=index_file_path, new_db=is_new_db, db_size=db_size)

    # Generate random query vector
    nq = 1
    query_vector = np.random.random((nq, 70)).astype(np.float32)

    # Search for the nearest neighbors of the query vector
    k = 5
    # Measure memory usage
    memory_usage = measure_memory_usage(vec_db.retrieve, query_vector, k)

    # Measure the time taken to search for the nearest neighbors

    start_time = timeit.default_timer()
    result = vec_db.retrieve(query_vector, k)
    end_time = timeit.default_timer()

    print(f"Time taken to search for the nearest neighbors: {end_time - start_time:.4f} seconds")
    print(f"Memory usage: {memory_usage[1]:.4f} MB")
    print(f"Average Time taken to search for the nearest neighbors: {(end_time - start_time)/nq} seconds")




    





