# A Statistical Perspective on Parallel Discovery of Conditional Functional Dependencies

> This code is the SCFDM_part of mining only constant CFDs

## Quick Start Guide
The code compiles with cmake. 
The algorithm takes three mandatory arguments (in input.txt)

1. the name of sub-tables in csv format
2. The minimum support threshold, specifying how often the discovered rules should occur in the data
3. The maximum size of the discovered rules, i.e., how many attributes are allowed to occur in any rule

**OpenMP parallel command：**

> 1.compile CMakeLists.txt
> 2.**Change the number of files and input directories** in input.txt and main.cpp
> 3.Go to the **cmake-build-debug directory** and run the command **cmake../**
> 3.# Start 30 threads (for example) with the following command:
> ulimit -s unlimited && mpiexec -n 30 -mca btl ^openib SCFDM_all
