import numpy as np
from memory_profiler import memory_usage
from scipy.spatial.distance import cdist
import os
import time
import random

def compute_recall_at_k(ground_truth: np.ndarray, faiss_results: np.ndarray, k: int) -> float:
    """
    Calculate recall for multiple queries.

    Parameters:
        ground_truth (np.ndarray): Ground truth indices of shape (n_queries, k).
        faiss_results (np.ndarray): FAISS result indices of shape (n_queries, k).
        k (int): Number of nearest neighbors to consider.

    Returns:
        float: Average recall across all queries.
    """
    total_recall = 0.0
    n_queries = ground_truth.shape[0]

    for i in range(n_queries):
        # Convert ground truth and FAISS results for each query into sets
        ground_truth_set = set(ground_truth[i, :k])
        faiss_result_set = set(faiss_results[i, :k])

        # Calculate recall for the current query
        recall = len(ground_truth_set & faiss_result_set) / k
        total_recall += recall

    # Return the average recall across all queries
    average_recall = total_recall / n_queries
    return average_recall

# Compute recall with ground truth being a list of indices of a single query vector
def compute_recall_at_k_single_query(ground_truth: np.ndarray, faiss_results: np.ndarray, k: int) -> float:
    """
    Calculate recall for a single query.

    Parameters:
        ground_truth (np.ndarray): Ground truth indices of shape (k,).
        faiss_results (np.ndarray): FAISS result indices of shape (k,).
        k (int): Number of nearest neighbors to consider.

    Returns:
        float: Recall for the single query.
    """
    # Convert ground truth and FAISS results for the query into sets
    ground_truth_set = set(ground_truth[:k])
    faiss_result_set = set(faiss_results[:k])

    # Calculate recall for the query
    recall = len(ground_truth_set & faiss_result_set) / k
    return recall

def measure_memory_usage(func, *args, **kwargs):
    """
    Measure memory usage during the execution of a function.

    Parameters:
    - func: The target function to measure.
    - *args: Positional arguments for the target function.
    - **kwargs: Keyword arguments for the target function.

    Returns:
    - result: The result of the target function.
    - memory_diff: The difference in memory usage before and after running the function.
    """
    # Record memory usage before execution
    mem_before = max(memory_usage())

    # Run the function and track memory usage during execution
    memory_values = memory_usage(proc=(func, args, kwargs), interval=1e-3)
    
    result = func(*args, **kwargs)
    
    memory_diff = max(memory_values) - mem_before

    return result, memory_diff


def measure_random_io_operations(duration=3, file_size=1024*1024*10, block_size=4096):
    """
    Measures the number of random I/O operations (reads and writes) that can be performed in a given duration.
    Args:
        duration (int): The duration in seconds to measure the I/O operations.
        file_size (int): The size of the file to simulate (default: 10MB).
        block_size (int): The size of each read/write operation (default: 4KB).
    Returns:
        int: The total number of I/O operations performed.
    """
    filename = "io_test_random.tmp"
    io_count = 0

    # Pre-allocate a file with random data
    with open(filename, "wb") as f:
        f.write(os.urandom(file_size))

    try:
        start_time = time.time()

        # Perform random reads and writes within the file
        with open(filename, "r+b") as f:
            while time.time() - start_time < duration:
                # Generate a random offset
                offset = random.randint(0, file_size - block_size)
                
                # Perform a random write
                f.seek(offset)
                f.write(os.urandom(block_size))
                f.flush()
                io_count += 1

                # Perform a random read
                f.seek(offset)
                f.read(block_size)
                io_count += 1

    finally:
        if os.path.exists(filename):
            os.remove(filename)

    return io_count


