This repository provides an efficient framework for Conditional Functional Dependency (CFD) discovery. The implementation is organized into two main modules:

### 1. Representation Tuple Sampling & Correlation Attribute Set Extraction

- **Directory**: `/correlation-and-sampling`
- **Description**: Focuses on the initial data preparation phase. It implements **representative tuple sampling** to reduce data scale while preserving statistical diversity. It also features **Attribute Correlation Set extraction** , utilizing a Transformer-based model to capture deep attribute dependencies through multi-epoch training and refined probing.

### 2. Sub-table Generation & Parallel CFD Mining on large datasets

- **Directory**: `/parallel`
- **Description**: Handles the high-performance mining phase. This module uses the extracted correlation sets to perform **vertical partitioning**, generating optimized sub-tables. It then executes **parallel CFD mining** across these sub-tables, significantly accelerating the discovery process on large-scale datasets such as RT-IoT2022.