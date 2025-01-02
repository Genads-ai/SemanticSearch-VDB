# Semantic Search Engine with Vectorized Databases

## Project Overview
This project implements a **Semantic Search Engine** that uses vectorized databases for efficient information retrieval based on vector space embeddings. The system is designed to handle large datasets and supports high-performance approximate nearest neighbor (ANN) searches.

### Objectives
- Build an indexing system capable of storing and retrieving vector embeddings.
- Optimize for retrieval speed, memory usage, and accuracy.
- Support up to 20 million records while adhering to strict performance constraints.

## Key Features
- **Indexing Mechanism**: Implements efficient indexing strategies for vector embeddings, leveraging IMI (Inverted Multi-Dimensional Index) and IVF-PQ (Inverted File with Product Quantization) techniques.
- **High Scalability**: Handles datasets ranging from 1M to 20M rows with minimal memory footprint.
- **Custom Retrieval Logic**: Efficiently retrieves the top-k most similar vectors using cosine similarity.
- **Optimization Techniques**: Includes multithreading, pruning, batching, and memory optimization for improved performance.

## System Requirements
| Dataset Size | Max RAM Usage (MB) | Max Retrieval Time (s) | Max Index Size (MB) |
|--------------|---------------------|-------------------------|----------------------|
| 1M Rows      | 20                 | 3                       | 50                   |
| 10M Rows     | 50                 | 6                       | 100                  |
| 15M Rows     | 50                 | 8                       | 150                  |
| 20M Rows     | 50                 | 10                      | 200                  |

## Implementation Details
### Indexing Approaches
1. **Inverted Multi-Dimensional Index (IMI)**:
   - Vectors are split into two subspaces and clustered using KMeans.
   - Inverted index structure associates vectors with centroid pairs.
   - Retrieval uses pruning and multithreaded processing for efficiency.

2. **Inverted File with Product Quantization (IVF-PQ)**:
   - Combines inverted indexing with product quantization for ANN search.
   - Suitable for smaller datasets (up to 1M rows).

### Optimization Strategies
- **Multithreading**: Parallel processing of vector batches.
- **Batching**: Manageable memory usage with reduced I/O overhead.
- **Pruning**: Early reduction of candidate vectors.
- **Memory Optimization**: Uses `np.float16` for computations.

## Evaluation Metrics
- **Accuracy**: Top-k recall based on cosine similarity.
- **Efficiency**: Low retrieval times and memory usage.
- **Scalability**: Ability to handle datasets up to 20M rows.

## Deliverables
1. Index files for datasets of sizes 1M, 10M, 15M, and 20M rows.
2. Python implementation for generating databases, building indices, and retrieving results.
3. Design document detailing the system architecture and optimization choices.

## Results
| Dataset Size | Score | Time (s) | RAM (MB) |
|--------------|-------|----------|----------|
| 1M           | 0.0   | 0.14     | 0.15     |
| 10M          | 0.0   | 0.95     | 0.77     |
| 15M          | 0.0   | 1.45     | 1.00     |
| 20M          | 0.0   | 2.06     | 1.24     |

## Usage Instructions
### Repository Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/farah-moh/vec_db
   cd vec_db
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Evaluation
Follow the steps in the provided evaluation notebook:
[Evaluation Notebook](https://colab.research.google.com/drive/1NDjJl623MuTBXJcVvtd09zIW-zF5sjIZ#scrollTo=hV2Nc_f8Mbqh)

## Authors
- Abdallah Ahmed Ali
- Ahmed Osama Helmy
- Aliaa Abdelaziz
- Omar Mahmoud Mohamed

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
