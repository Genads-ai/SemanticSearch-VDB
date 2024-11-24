import numpy as np


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