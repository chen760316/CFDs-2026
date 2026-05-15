# A Statistical Perspective on Parallel Discovery of Conditional Functional Dependencies

> This code is the SCFDM_all of mining both constant and variable CFDs

## Quick Start Guide
The code compiles with cmake. 
The algorithm takes five mandatory arguments and one optional as input (in input.txt):

1. sub-tables in csv format

2. The minimum support threshold

3. The confidence leeway threshold

4. The maximum antecedent size of the discovered CFDs

5. The chosen algorithm(By default, the option FD-First-DFS-dfs is chosen, as it is typically the fastest.)

   ```
   * Integrated-BFS (CTane)
   * Integrated-DFS
   * Itemset-First-BFS-bfs
   * Itemset-First-BFS-dfs
   * Itemset-First-DFS-bfs
   * Itemset-First-DFS-dfs
   * FD-First-BFS-bfs
   * FD-First-BFS-dfs
   * FD-First-DFS-bfs
   * FD-First-DFS-dfs
   ```

OpenMP parallel command：

> 1.compile CMakeLists.txt
> 2.**Change the number of files and input directories** in input.txt and main.cpp
> 3.Go to the **cmake-build-debug directory** and run the command **cmake../**
> 3.# Start 30 threads (for example) with the following command:
> ulimit -s unlimited && mpiexec -n 30 -mca btl ^openib SCFDM_all
