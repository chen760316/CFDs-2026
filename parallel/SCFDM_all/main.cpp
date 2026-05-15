#include <fstream>
#include <iostream>
#include <cstring>
#include <chrono>
#include "util/output.h"
#include "util/stringutil.h"
#include "data/databasereader.h"
#include "algorithms/cfddiscovery.h"
#include <string>
#include <sstream>
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <thread>
#include <unistd.h>

int main() {
    MPI_Init(nullptr, nullptr);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int num_files = 1;
    std::string line;
    std::string filename;
    std::vector<std::string> filenames;
    std::vector<std::string> parameters;
    int supp;
    double conf;
    int maxSize;
//    std::string strategy;
    const char* strategy;
    int count = 0;
    std::ifstream file(R"(./input.txt)");
    if (!file) {
        std::cout << "Can not open the file" << std::endl;
        return 1;
    }
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        std::cout << "Current working directory: " << cwd << std::endl;
    }
    std::getline(file, line);
    std::stringstream ss(line);
    while (ss >> filename) {
        filenames.push_back(filename);
    }
    while (std::getline(file, line)) {
        parameters.push_back(line);
    }
    for (const auto &param: parameters) {
        if (count == 0) {
            supp = std::stoi(param);
        } else if (count == 1) {
            conf = std::stod(param);
        } else if (count == 2) {
            maxSize = std::stoi(param) + 1;
        } else if (count == 3) {
            strategy = param.c_str();
        }
        count++;
    }
    int files_per_process = num_files / size;
    int start_file = rank * files_per_process;
    int end_file = (rank == size - 1) ? num_files : start_file + files_per_process;
    auto start = std::chrono::high_resolution_clock::now();
    omp_set_num_threads(1);
#pragma omp parallel for schedule (dynamic, 1)
    for (int i = start_file; i < end_file; ++i) {
//        printf("I am process %d, thread %d, i=%d\n", rank, omp_get_thread_num(), i);
        std::string fullPath = "./datasets/" + filenames[i];
        std::ifstream dataFile(fullPath);
        if (!dataFile.good()) {
            std::cout << "[ERROR] File not found: " << fullPath << std::endl;
        }
        Database db = DatabaseReader::fromTable(dataFile, ',');
        CFDDiscovery cfdd(db);
        if (strcmp(strategy, "FD-First-DFS-dfs") == 0) {
            cfdd.fdsFirstDFS(supp, maxSize, SUBSTRATEGY::DFS, conf);
        }
        else if (strcmp(strategy, "FD-First-DFS-bfs") == 0) {
            cfdd.fdsFirstDFS(supp, maxSize, SUBSTRATEGY::BFS, conf);
        }
        else if (strcmp(strategy, "FD-First-BFS-dfs") == 0) {
            cfdd.fdsFirstBFS(supp, maxSize, SUBSTRATEGY::DFS, conf);
        }
        else if (strcmp(strategy, "FD-First-BFS-bfs") == 0) {
            cfdd.fdsFirstBFS(supp, maxSize, SUBSTRATEGY::BFS, conf);
        }
        else if (strcmp(strategy, "Itemset-First-DFS-dfs") == 0) {
            cfdd.itemsetsFirstDFS(supp, maxSize, SUBSTRATEGY::DFS, conf);
        }
        else if (strcmp(strategy, "Itemset-First-DFS-bfs") == 0) {
            cfdd.itemsetsFirstDFS(supp, maxSize, SUBSTRATEGY::BFS, conf);
        }
        else if (strcmp(strategy, "Itemset-First-BFS-dfs") == 0) {
            cfdd.itemsetsFirstBFS(supp, maxSize, SUBSTRATEGY::DFS, conf);
        }
        else if (strcmp(strategy, "Itemset-First-BFS-bfs") == 0) {
            cfdd.itemsetsFirstBFS(supp, maxSize, SUBSTRATEGY::BFS, conf);
        }
        else if (strcmp(strategy, "Integrated-DFS") == 0) {
            cfdd.integratedDFS(supp, maxSize, conf);
        }
        else if (strcmp(strategy, "Integrated-BFS") == 0) {
            cfdd.ctane(supp, maxSize, conf);
        }
        CFDList cfds = cfdd.getCFDs();
        // 不打印CFD
        for (const auto& c : cfds) {
            Output::printCFD(c, db);
        }
        std::cout << "Mined " << cfds.size() << "CFDs" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Mined cfds in " << time << " milliseconds" << std::endl;
    return 0;
}