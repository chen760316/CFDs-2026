## A Statistical Perspective on Parallel Discovery of Conditional Functional Dependencies

> Under this project, implementations are provided for the correlated attribute sets extraction algorithm AttrFinder, the representative sampling algorithm RepSampler, the embedding process, sub-table generation, supplementary sampling, as well as code for testing precision and recall, involving corresponding implementation algorithms and utility classes.

## Quick Start Guide

The code compiles with Python 3.7. We have set default parameters in the algorithm. Below are the environments and their respective version numbers that you need to configure:

```
# Configure the environment with one click
pip install -r requirements.txt
# Run the code with one click
python xx.py
```

## Project structure

To make it easier for readers to understand the roles and execution of each algorithm in the article, we have divided the main functional modules into sub folders and placed independently executable Python files within them.

Specifically, the folder divisions are as follows:

> 1. The `correlation_extraction` folder implements corresponding algorithms for correlation column extraction, embedding algorithms, and CFDs recall rate calculation.
> 4. The `sampling` folder implements multiple versions of representative sampling, subtable generation, algorithms for CFDs accuracy calculation, multiple versions of CFDs rapid verification modules, and a resampling module.
> 5. The `tests` folder provides test cases for some pre-trained models.
> 6. The `utils` folder contains implementations of utility classes for all the functionalities mentioned above.

