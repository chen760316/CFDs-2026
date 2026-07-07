# Fast Discovery of Conditional Functional Dependencies via Transformer-Guided Relation Partitioning

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. System Architecture](#2-system-architecture)
- [3. Repository Structure](#3-repository-structure)
- [4. Environment & Dependencies](#4-environment--dependencies)
- [5. Module I: Correlated Attribute Set Extraction (AttrFinder)](#5-module-i-correlated-attribute-set-extraction-attrfinder)
- [6. Module II: Representative Tuple Sampling (RepSampler)](#6-module-ii-representative-tuple-sampling-repsampler)
- [7. Module III: Sub-table Generation](#7-module-iii-sub-table-generation)
- [8. Module IV: Parallel CFD Mining (SCFDM)](#8-module-iv-parallel-cfd-mining-scfdm)
- [9. Sampling Lower Bounds (Theorem 1 & 2)](#9-sampling-lower-bounds-theorem-1--2)
- [10. Adaptive Hyperparameter Tuning](#10-adaptive-hyperparameter-tuning)
- [11. Evaluation Framework](#11-evaluation-framework)
- [12. Ablation Studies](#12-ablation-studies)
- [13. End-to-End Pipeline Orchestration](#13-end-to-end-pipeline-orchestration)
- [14. Experiment Reproduction Guide](#14-experiment-reproduction-guide)
- [15. Datasets](#15-datasets)
- [16. Algorithm Reference](#16-algorithm-reference)
- [17. Citation](#17-citation)

---

## 1. Overview

This repository provides the complete implementation of **SCFDM** (Statistical CFD Miner), a framework for fast discovery of Conditional Functional Dependencies (CFDs) on large-scale relational data. The core idea is to leverage a **Transformer-guided relation partitioning** strategy that:

1. **Extracts correlated attribute sets** using a Transformer-based masked-attribute probing model (AttrFinder), replacing exhaustive pairwise candidate enumeration with learned attention-driven dependency capture.
2. **Reduces data volume** via statistically guaranteed representative tuple sampling (RepSampler), with sample size lower bounds derived from variance-augmented Hoeffding-Bennstein inequalities.
3. **Partitions the relation** vertically into compact sub-tables based on the discovered correlation sets, enabling independent parallel mining.
4. **Mines CFDs in parallel** across sub-tables using MPI + OpenMP, achieving near-linear speedup on multi-core clusters.

The framework supports discovery of both **constant CFDs** (e.g., `(A=a, B=b) => C=c`) and **variable CFDs** (e.g., `(A, B) => C` where pattern values are wildcards).

### Key Contributions

| Component | Technique | Paper Section |
|---|---|---|
| AttrFinder | Transformer encoder with masked-attribute reconstruction | §6 |
| RepSampler | MinHash-LSH diversity-first sampling with Theorem 2 bounds | §7 |
| Sub-table generation | Vertical partitioning guided by correlation sets Ψ | §8 |
| SCFDM (C++) | MPI + OpenMP parallel CFD discovery with 10 search strategies | §8 |
| Adaptive thresholds | Normalized Shannon entropy + negative-feedback hyperparameter tuning | §7.3, Appendix B |
| Sampling bounds | Variance-augmented Hoeffding-Bennstein piecewise bound | §7.1 |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Relation (CSV)                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │   Stage 1: AttrFinder (§6)      │
          │   Transformer masked probing    │
          │   → Correlated sets Ψ           │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │   Stage 2: Sampling Bound (§7.1)│
          │   Theorem 2 piecewise bound     │
          │   → Minimum sample size N       │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │   Stage 3: RepSampler (§7)      │
          │   MinHash-LSH diversity sampling│
          │   → Representative sample S     │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │   Stage 4: Sub-table Gen (§8)   │
          │   Project S onto Ψ              │
          │   → Sub-tables {T₁,...,Tₖ}     │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │   Stage 5: Parallel Mining (§8) │
          │   MPI + OpenMP CFD discovery    │
          │   → CFD set Σ̂                   │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │   Stage 6: Evaluation (§9.1)    │
          │   Precision / Recall / F1       │
          └─────────────────────────────────┘
```

---

## 3. Repository Structure

```
code-for-paper/
│
├── correlation-and-sampling/          # Python: AttrFinder + RepSampler + Evaluation
│   ├── correlation_extraction/        #   Correlated attribute set extraction
│   │   ├── correlation_extraction.py          # Sentence-Transformer + PCA + Lasso pipeline
│   │   ├── correlation_extraction_tf.py       # Transformer (AttrFinder) single-GPU
│   │   ├── correlation_extraction_tf_multi.py # Transformer multi-epoch probing
│   │   ├── correlation_extraction_tf_gpu.py   # GPU-optimized variant
│   │   ├── correlation_extraction_tf_multi_gpu.py
│   │   ├── correlation_extraction_GCFD.py     # GCFD-specific variant
│   │   ├── configuration.py                   # Adaptive Shannon entropy threshold engine
│   │   ├── adaptive_thresholds.py             # Hyperparameter negative-feedback pipeline
│   │   └── storage/                           # Cached embeddings (.pkl)
│   │
│   ├── sampling/                      #   Representative sampling & verification
│   │   ├── representative_tuple_sampling.py   # RepSampler (MinHash-LSH)
│   │   ├── k_means_with_dbscan.py             # K-Means + DBSCAN sampling variant
│   │   ├── k_means_with_dbscan_v2.py          # Optimized K-Means variant
│   │   ├── dbscan_sample.py                   # DBSCAN-only sampling
│   │   ├── fast_dbscan_sampling.py            # Fast DBSCAN sampling
│   │   ├── k_means_sample.py                  # K-Means-only sampling
│   │   ├── rep_with_kmeans.py                 # Rep + K-Means hybrid
│   │   ├── generate_sub_tables.py             # Sub-table CSV generation
│   │   ├── verify.py                          # CFD verification (support/confidence)
│   │   ├── verify_v2.py                       # Optimized verification
│   │   ├── verify_v3.py                       # Multi-process parallel verification
│   │   ├── sample_utils.py                    # Shared sampling utilities
│   │   ├── lhs_rhs.py                         # LHS/RHS pattern extraction
│   │   ├── same_cfds.py                       # CFD deduplication
│   │   └── GCFD/                              # GCFD-specific sampling outputs
│   │
│   ├── sentence_transformers/         #   Embedded SentenceTransformer library
│   │   ├── SentenceTransformer.py             # Core model wrapper
│   │   ├── cross_encoder/                     # Cross-encoder modules
│   │   ├── models/                            # Model definitions
│   │   ├── losses/                            # Training loss functions
│   │   ├── readers/                           # Data readers
│   │   ├── evaluation/                        # STB evaluation tools
│   │   └── util.py                            # Utility functions
│   │
│   ├── supplemental/                  #   Supplementary theory & experiments
│   │   ├── sampling_bound/
│   │   │   ├── sampling_bound.py              # Theorem 1 & 2 lower bounds
│   │   │   └── run_bound_comparison.py        # Bound comparison experiments
│   │   ├── evaluation/
│   │   │   ├── cfd_parser.py                  # CFD parsing & normalization
│   │   │   ├── metrics.py                     # Precision/Recall/F1
│   │   │   └── evaluate_cfds.py               # End-to-end evaluation
│   │   └── experiments/
│   │       ├── run_pipeline.py                # Full pipeline orchestration
│   │       ├── run_scalability.py             # 6-dimension scalability sweep
│   │       └── run_parameter_sensitivity.py   # δ̂ parameter sensitivity
│   │
│   ├── ablation/                      #   Ablation study baselines
│   │   ├── correlation/
│   │   │   ├── correlation_extraction_XGBoost_multi.py  # XGBoost baseline
│   │   │   └── correlation_extraction_kamino_multi.py   # Kamino baseline
│   │   └── sampling/
│   │       ├── random_sampling.py             # Random sampling baseline
│   │       └── Stratified_sampling.py         # Stratified sampling baseline
│   │
│   ├── utils/                         #   Shared utility functions
│   │   ├── utils_correlation.py               # PCA, Pearson, Lasso, clustering
│   │   ├── utils_correlation_plus.py          # Extended correlation utils
│   │   ├── utils_correlation_GCFD.py          # GCFD-specific utils
│   │   ├── utils_GCFDS.py / _v2.py / _v3.py  # GCFD utility versions
│   │   ├── utils_large.py / _v6.py / _row.py  # Large-scale dataset utils
│   │   ├── utils_representative_sampling.py   # RepSampler utilities
│   │   ├── utils_rep_with_kmeans.py           # Rep+KMeans utilities
│   │   └── utils_initial.py                   # Initial table processing
│   │
│   ├── tests/                         #   Unit tests for SentenceTransformer
│   │   ├── test_compute_embeddings.py
│   │   ├── test_cross_encoder.py
│   │   ├── test_evaluator.py
│   │   ├── test_multi_process.py
│   │   └── ...
│   │
│   ├── datasets_synthetic/datasets/   #   Synthetic Bayesian network data
│   │   ├── alarm.csv
│   │   ├── barley.csv
│   │   └── barley_long.csv
│   │
│   ├── requirements.txt               #   Python dependencies
│   └── README.md                      #   Module-level README
│
├── parallel/                          # C++: Parallel CFD Mining (SCFDM)
│   ├── SCFDM_all/                     #   Full CFD discovery (constant + variable)
│   │   ├── main.cpp                           # MPI/OpenMP entry point
│   │   ├── CMakeLists.txt                     # Build configuration
│   │   ├── input.txt                          # Runtime parameters
│   │   ├── algorithms/
│   │   │   ├── cfddiscovery.cpp / .h          # 10 search strategies
│   │   │   ├── partitiontable.cpp / .h        # Partition-based tid lists
│   │   │   ├── minernode.cpp / .h             # Mining tree nodes
│   │   │   └── generatorstore.h               # Generator set storage
│   │   ├── data/
│   │   │   ├── database.cpp / .h              # In-memory relation
│   │   │   ├── databasereader.cpp / .h        # CSV reader
│   │   │   ├── cfd.cpp / .h                   # CFD data structure
│   │   │   └── types.cpp / .h                 # Type definitions
│   │   ├── util/
│   │   │   ├── setutil.cpp / .h               # Set operations
│   │   │   ├── prefixtree.h                   # Prefix tree for candidates
│   │   │   ├── output.h                       # CFD output formatting
│   │   │   └── stringutil.cpp / .h            # String utilities
│   │   ├── datasets/                          # Test sub-tables
│   │   └── data/                              # Test relations
│   │
│   └── SCFDM_part/                    #   Constant CFD discovery only
│       ├── main.cpp                           # MPI/OpenMP entry point
│       ├── CMakeLists.txt
│       ├── input.txt
│       ├── algorithms/
│       │   ├── ccfdminer.cpp / .h             # Constant CFD miner
│       │   ├── minernode.cpp / .h
│       │   └── genmapentry.h                  # Generator map
│       └── util/ / data/ / datasets/
│
├── Fast Discovery of Conditional Functional Dependencies
    via Transformer-Guided Relation Partitioning (full version).pdf
│
└── README.md                          # Project-level README
```

---

## 4. Environment & Dependencies

### 4.1 Python Environment (correlation-and-sampling/)

```bash
# Python 3.7+
cd correlation-and-sampling
pip install -r requirements.txt
```

**Key dependencies:**

| Package | Version | Purpose |
|---|---|---|
| `torch` | 1.7.0 | Transformer model (AttrFinder) |
| `tensorflow` | 2.11.0 | Optional GPU acceleration |
| `sentence-transformers` | 2.2.2 | Column embedding (all-MiniLM-L6-v2) |
| `transformers` | 4.30.2 | HuggingFace model support |
| `scikit-learn` | 1.0.2 | PCA, clustering, metrics |
| `pandas` | 1.1.5 | Data manipulation |
| `numpy` | — | Numerical computation |
| `scipy` | 1.7.3 | Statistical functions |
| `datasketch` | — | MinHash-LSH for RepSampler |
| `xgboost` | — | Ablation baseline |
| `faiss-cpu` | 1.7.4 | Fast nearest-neighbor search |
| `ray` | 2.6.1 | Distributed computing (optional) |

### 4.2 C++ Environment (parallel/)

**Build tools:**
- CMake ≥ 2.6
- C++ compiler supporting C++11 (`-std=c++0x`)
- Boost ≥ 1.40 (`program_options`)
- MPI (OpenMPI or MPICH)
- OpenMP
- LLVM (for `llvm_map_components_to_libnames`)

```bash
# Build SCFDM_all (constant + variable CFDs)
cd parallel/SCFDM_all
mkdir -p cmake-build-debug && cd cmake-build-debug
cmake ..
make

# Build SCFDM_part (constant CFDs only)
cd parallel/SCFDM_part
mkdir -p cmake-build-debug && cd cmake-build-debug
cmake ..
make
```

### 4.3 Hardware Recommendations

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 8 cores | 28–112 cores (MPI cluster) |
| RAM | 64 GB | 128 GB+ (for large datasets) |
| GPU | Optional | NVIDIA GPU with ≥ 32 GB VRAM |
| Storage | 50 GB | 100 GB+ (for large datasets & embeddings) |

---

## 5. Module I: Correlated Attribute Set Extraction (AttrFinder)

> **Paper Section:** §6, Algorithm 1
> **Directory:** `correlation-and-sampling/correlation_extraction/`

### 5.1 Overview

AttrFinder replaces the traditional exhaustive pairwise candidate enumeration with a **Transformer-based masked-attribute probing** approach. The key insight is that if attribute Y can be reconstructed from the remaining attributes when Y's embedding is masked to zero, then the remaining attributes carry dependency information about Y.

### 5.2 Model Architecture

```
AttrFinder (nn.Module)
├── Embedding layer: per-attribute nn.Embedding(cardinality, d_model)
├── Transformer Encoder: nn.TransformerEncoderLayer × num_layers
│   ├── d_model = 128 (default)
│   ├── nhead = 4
│   └── num_layers = 2
└── Reconstruction head: per-attribute nn.Linear(d_model, cardinality)
```

**Forward pass:**
1. Each attribute value is embedded into a `d_model`-dimensional vector.
2. The target attribute's embedding is zeroed out (masked probing).
3. The Transformer encoder processes the concatenated embedding sequence.
4. A linear head reconstructs the masked attribute from the Transformer output.

### 5.3 Variants

| File | Description | Use Case |
|---|---|---|
| `correlation_extraction.py` | Sentence-Transformer + PCA + Lasso + Hierarchical clustering | Small/medium datasets, multi-method fusion |
| `correlation_extraction_tf.py` | Single-GPU Transformer (d_model=32) | Quick prototyping |
| `correlation_extraction_tf_multi.py` | Multi-epoch Transformer (d_model=64, 12 epochs) | Production, rule refinement |
| `correlation_extraction_tf_gpu.py` | GPU-optimized single-GPU | Large datasets |
| `correlation_extraction_tf_multi_gpu.py` | Multi-GPU multi-epoch | Very large datasets (RT-IoT2022) |
| `correlation_extraction_GCFD.py` | GCFD-specific variant | Generalized CFD discovery |

### 5.4 Usage

```python
from correlation_extraction_tf_multi import run_full_data_pipeline

run_full_data_pipeline(
    input_file='datasets/RT_IOT2022.csv',
    sets_output='output/correlated_sets.txt'
)
# Output format (one rule per line):
# {attr1, attr2} -> target_attr
```

### 5.5 Adaptive Threshold Engine

The `configuration.py` module implements the **normalized Shannon entropy adaptive audit** (§7.3, Appendix A.1):

1. Extract the global average attention matrix Ā from the Transformer.
2. Compute normalized Shannon entropy H for each attribute row.
3. Dynamically compute the elastic retention threshold: `γ = max(γ₀, 1 - avg(H))`.
4. For each target attribute Y, select the minimal antecedent set X such that cumulative attention weight ≥ γ and individual weight ≥ θ_att = z/m.

```python
from configuration import extract_adaptive_correlated_sets

partitions, gamma = extract_adaptive_correlated_sets(
    model, base_model, gamma_0=0.85, z=3
)
```

---

## 6. Module II: Representative Tuple Sampling (RepSampler)

> **Paper Section:** §7, Algorithm 2
> **Directory:** `correlation-and-sampling/sampling/`

### 6.1 Overview

RepSampler reduces the data volume while preserving statistical diversity using a **MinHash-LSH diversity-first selection** strategy:

1. **Stream scan**: Each tuple is hashed using MinHash (128 permutations) to produce a compact signature.
2. **Bucketing**: Tuples with similar signatures are grouped into buckets (LSH buckets using the first 4 hash values as bucket keys).
3. **Diversity-first selection**: Buckets are sorted by size (ascending). A round-robin selection picks one tuple from each bucket, ensuring diverse coverage of the data distribution.
4. **Two-pass I/O**: Only row indices are stored in memory during the first pass; actual data is extracted in the second pass, enabling processing of datasets larger than RAM.

### 6.2 Usage

```python
from representative_tuple_sampling import RepSamplerProcessor

processor = RepSamplerProcessor('datasets/large_dataset.csv')
sampled_df = processor.run_repsampler(N_bound=8400, num_perm=128)
sampled_df.to_csv('output/sampled.csv', index=False)
```

### 6.3 Sampling Variants

| File | Method | Description |
|---|---|---|
| `representative_tuple_sampling.py` | MinHash-LSH | Main RepSampler implementation |
| `k_means_with_dbscan.py` | K-Means + DBSCAN | Cluster-based representative sampling |
| `k_means_with_dbscan_v2.py` | K-Means + DBSCAN v2 | Optimized version |
| `dbscan_sample.py` | DBSCAN only | Density-based sampling |
| `fast_dbscan_sampling.py` | Fast DBSCAN | Accelerated DBSCAN sampling |
| `k_means_sample.py` | K-Means only | Centroid-based sampling |
| `rep_with_kmeans.py` | RepSampler + K-Means | Hybrid approach |

---

## 7. Module III: Sub-table Generation

> **Paper Section:** §8
> **Directory:** `correlation-and-sampling/sampling/generate_sub_tables.py`

After AttrFinder produces correlated attribute sets Ψ = {(X₁→Y₁), ..., (Xₖ→Yₖ)}, the original relation is vertically partitioned into sub-tables. Each sub-table Tᵢ contains only the columns in Xᵢ ∪ {Yᵢ}, dramatically reducing the search space for CFD discovery.

```python
# Input: correlated_sets.txt (from AttrFinder)
# Output: part0.csv, part1.csv, ..., partk.csv

# Each line in correlated_sets.txt:
# {col1, col2} -> col3
# → Sub-table contains [col1, col2, col3]
```

The `generate_sub_tables.py` script reads the correlation sets and the sampled data, then projects the relevant columns into individual CSV files for parallel mining.

---

## 8. Module IV: Parallel CFD Mining (SCFDM)

> **Paper Section:** §8, Algorithm 3
> **Directory:** `parallel/`

### 8.1 SCFDM_all — Full CFD Discovery (Constant + Variable)

**Entry point:** `parallel/SCFDM_all/main.cpp`

Supports **10 search strategies**, selectable via `input.txt`:

| Strategy | Description |
|---|---|
| `Integrated-BFS` | CTane algorithm (BFS level-wise) |
| `Integrated-DFS` | Integrated DFS with free itemset pruning |
| `Itemset-First-BFS-bfs` | Itemset-first BFS, pattern BFS sub-strategy |
| `Itemset-First-BFS-dfs` | Itemset-first BFS, pattern DFS sub-strategy |
| `Itemset-First-DFS-bfs` | Itemset-first DFS, pattern BFS sub-strategy |
| `Itemset-First-DFS-dfs` | Itemset-first DFS, pattern DFS sub-strategy |
| `FD-First-BFS-bfs` | FD-first BFS, pattern BFS sub-strategy |
| `FD-First-BFS-dfs` | FD-first BFS, pattern DFS sub-strategy |
| `FD-First-DFS-bfs` | FD-first DFS, pattern BFS sub-strategy |
| `FD-First-DFS-dfs` | FD-first DFS, pattern DFS sub-strategy (default, fastest) |

**input.txt format:**
```
<sub-table filenames, space-separated>
<minimum support threshold>
<confidence threshold>
<max antecedent size>
<strategy name>
```

**Example:**
```
CENSUS42-10000.csv
1994
1
1
FD-First-DFS-dfs
```

**Run command:**
```bash
# Compile
cd parallel/SCFDM_all/cmake-build-debug
cmake .. && make

# Execute with MPI (30 processes)
ulimit -s unlimited && mpiexec -n 30 -mca btl ^openib SCFDM_all
```

### 8.2 SCFDM_part — Constant CFD Discovery Only

**Entry point:** `parallel/SCFDM_part/main.cpp`

Uses the `CCFDMiner` algorithm for mining only **constant CFDs** (where all pattern values are concrete, no wildcards). This is faster but discovers fewer rules.

**input.txt format:**

```
<sub-table filenames, space-separated>
<minimum support threshold>
<max rule size>
```

**Example:**

```
part0.csv part1.csv part2.csv part3.csv part4.csv
1200
100
```

### 8.3 Parallel Architecture

- **MPI**: Distributes sub-tables across processes (one sub-table per process per iteration).
- **OpenMP**: Each process uses OpenMP threads for intra-sub-table parallelism.
- **Dynamic scheduling**: `#pragma omp parallel for schedule(dynamic, 1)` ensures load balancing across heterogeneous sub-table sizes.
- **Stack size**: `ulimit -s unlimited` is required for deep recursion in DFS strategies.

### 8.4 CFD Output Format

The C++ miners output CFDs in the following format (parsed by `verify.py` and `cfd_parser.py`):

```
[A, B] => C, (a1, b1 || c1)
```

Where:
- `[A, B]` — LHS attributes
- `C` — RHS attribute
- `(a1, b1 || c1)` — LHS pattern values `||` RHS pattern value
- `_` denotes a variable (wildcard) pattern

---

## 9. Sampling Lower Bounds (Theorem 1 & 2)

> **Paper Section:** §7.1
> **File:** `correlation-and-sampling/supplemental/sampling_bound/sampling_bound.py`

### 9.1 Theorem 1 — Variance-Augmented Hoeffding-Bennstein Bound

$$N = \frac{\hat{p}(1-\hat{p}) \cdot \ln\!\bigl(\frac{2}{1-\eta}\bigr)}{2\varepsilon^2}$$

Tighter than both Chebyshev and Bennett bounds, leveraging the variance of the Bernoulli indicator.

### 9.2 Theorem 2 — Piecewise Tightest Bound

Three regimes based on the relationship between p̂ and the Chernoff ratio λ:

| Regime | Condition | Bound |
|---|---|---|
| A (variance-augmented) | p̂ ≤ p̂₁ or p̂ > p̂₂, with 1/(24λ) ≥ 1 | Theorem 1 |
| B (hybrid Chernoff) | p̂₁ < p̂ ≤ p̂₂ | N = (3/2)·ln(2/(1-η)) / ε² |
| C (scaled variance) | 1/(24λ) < 1 | Theorem 1 / (1 − 1/(24λ)) |

Where:
- p̂₁ = (1 − √(1 − 24λ)) / 2
- p̂₂ = (1 + √(1 − 24λ)) / 2

### 9.3 Data-Driven p̂ Estimation

```python
from supplemental.sampling_bound.sampling_bound import p_hat_estimate, theorem2_bound

# Estimate p̂ from data
p_hat = p_hat_estimate(df, delta_hat=2e-3)

# Compute minimum sample size
N = theorem2_bound(p_hat, eps=5e-4, eta=0.9, lam=0.05)
```

### 9.4 Bound Comparison

```python
from supplemental.sampling_bound.sampling_bound import compare_bounds

bounds = compare_bounds(p_hat=0.01, eps=5e-4, eta=0.9, lam=0.05)
# Returns: {'variance_augmented': ..., 'chebyshev': ..., 'bennett': ...,
#           'hybrid_chernoff': ..., 'theorem2': ...}
```

---

## 10. Adaptive Hyperparameter Tuning

> **Paper Section:** §7.3, Appendix B
> **File:** `correlation-and-sampling/correlation_extraction/adaptive_thresholds.py`

The system implements a **negative-feedback control pipeline** that calibrates structural partitioning parameters based on a 10% data-slice probe:

### Two-Category Hyperparameter System

**Category 1 — Discovery Guarantee Parameters:**
- β (recall guarantee) — target recall rate, default 0.90
- ε (sampling error) — additive support error, default 0.005
- τ (concentration switch) — inequality switching coefficient, default 0.05

**Category 2 — Structural Partitioning Parameters:**
- z (significance threshold) — default 3, raised to 5 under noise
- γ₀ (retention density) — default 0.85, tightened to 0.80 under noise

### Feedback Logic

```
Probe 10% slice → generate preview partitions
├── If sub-tables fully loaded (m sub-tables, max size > 7):
│   → Raise z from 3 → 5, tighten γ₀ from 0.85 → 0.80
├── If zero sub-tables generated:
│   → Lower z from 3 → 2
└── Otherwise:
    → Maintain baseline configuration
```

### Adaptive Scaling

When the schema has > 50 attributes, ε is automatically scaled to prevent memory overflow:

```python
if m > 50:
    epsilon = 0.01  # doubled from 0.005
```

---

## 11. Evaluation Framework

> **Paper Section:** §9.1
> **Directory:** `correlation-and-sampling/supplemental/evaluation/`

### 11.1 CFD Parsing & Normalization

**File:** `cfd_parser.py`

Parses CFD output from the C++ miners and normalizes for set comparison:

```python
from supplemental.evaluation.cfd_parser import load_cfds, CFD

# Load mined CFDs
mined_cfds = load_cfds('mined_cfds.txt')
# Each CFD: CFD(lhs_attrs=('A','B'), rhs_attr='C', lhs_vals=('a1','b1'), rhs_val='c1')

# Comparison is order-invariant:
# [A,B]=>C,(a,b||c)  ==  [B,A]=>C,(b,a||c)
```

### 11.2 Metrics

**File:** `metrics.py`

$$P(\hat{\Sigma}, \Sigma) = \frac{|\hat{\Sigma} \cap \Sigma|}{|\hat{\Sigma}|}, \quad R(\hat{\Sigma}, \Sigma) = \frac{|\hat{\Sigma} \cap \Sigma|}{|\Sigma|}, \quad F_1 = \frac{2PR}{P+R}$$

```python
from supplemental.evaluation.metrics import evaluate

result = evaluate(mined_cfds, ground_truth_cfds)
print(f"Precision={result.precision:.4f}  Recall={result.recall:.4f}  F1={result.f1:.4f}")
```

### 11.3 CFD Verification

**Files:** `sampling/verify.py`, `verify_v2.py`, `verify_v3.py`

Verifies mined CFDs against the full dataset by checking support and confidence thresholds:

- `verify.py` — Single-process verification
- `verify_v2.py` — Optimized with pandas query
- `verify_v3.py` — Multi-process parallel verification using `multiprocessing.Pool`

---

## 12. Ablation Studies

> **Directory:** `correlation-and-sampling/ablation/`

### 12.1 Correlation Extraction Baselines

| File | Method | Description |
|---|---|---|
| `correlation/correlation_extraction_XGBoost_multi.py` | XGBoost | Feature importance-based attribute correlation |
| `correlation/correlation_extraction_kamino_multi.py` | Kamino | Alternative correlation extraction |
| `correlation_extraction_pearson_multi.py` | Pearson | Attribute correlation extraction |

**XGBoost baseline**: For each attribute Y, train an XGBoost model with all other attributes as features. Extract correlated attributes based on feature importance scores above threshold τ_c.

### 12.2 Sampling Baselines

| File | Method | Description |
|---|---|---|
| `sampling/random_sampling.py` | Random | Uniform random sampling without replacement |
| `sampling/Stratified_sampling.py` | Stratified | Stratified sampling based on attribute value distributions |

---

### 12.3 Stage Breakdown

| Stage | Module | Output |
|---|---|---|
| 1. AttrFinder | correlation_extraction_tf_multi.py | correlated_sets.txt |
| 2. Sampling Bound | sampling_bound.py | Minimum N |
| 3. RepSampler | representative_tuple_sampling.py | sampled.csv |
| 4. Sub-table Gen | generate_sub_tables.py | subtables/part*.csv |
| 5. Parallel Mining | SCFDM_all (C++) | mined CFDs |
| 6. Evaluation | metrics.py | Precision/Recall/F1 |

---

## 13. Datasets

### 13.1 Supported Datasets

We recommend the following datasets (configure paths in each script's `__main__`):

| Dataset | Size | Attributes | Characteristics |
|---|---|---|---|
| RT-IoT2022 | ~123K rows | 12 | IoT network traffic, multi-class |
| Adult (Census) | ~32K rows | 15 | Demographic data |
| Abalone | ~4K rows | 9 | Physical measurements |
| Bank Marketing | ~45K rows | 17 | Marketing campaign data |
| Census Income (KDD) | ~100K rows | 42 | Large-scale census |
| Flights (2015) | ~1M rows | 31 | Flight delays (large-scale) |
| Crop Mapping | ~75K rows | 148 | High-dimensional agricultural |
| CENSUS42 | 10K rows | 42 | Census subset |
| Mushroom | ~8K rows | 23 | Fungi characteristics |
| Nursery | ~13K rows | 9 | School application ranking |
| Contraceptive | ~1.5K rows | 10 | Demographic survey |
| Letter Recognition | 20K rows | 17 | Image features |

### 13.2 Synthetic Bayesian Network Data

Located in `correlation-and-sampling/datasets_synthetic/datasets/`:

| Dataset | Source Network |
|---|---|
| `alarm.csv` | ALARM medical diagnostic network |
| `barley.csv` | Barley disease network |
| `barley_long.csv` | Extended barley network |

### 13.3 Data Preprocessing

Utility scripts for data preparation:

```bash
# Convert TXT to CSV
python utils/txt_to_csv.py

# Remove columns from CSV
python utils/remove_columns_from_csv.py

# Remove rows from CSV
python utils/remove_lines_from_csv.py

# Remove empty lines
python utils/remove_empty_lines.py
```
