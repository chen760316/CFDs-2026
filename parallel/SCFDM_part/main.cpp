#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <chrono>
#include <omp.h>
#include <mpi.h>
#include "data/tabulardatabase.h"
#include "algorithms/ccfdminer.h"
#include <vector>
#include <chrono>
#include <future>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <unistd.h>

int main() {
    MPI_Init(nullptr, nullptr);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int num_files = 5;
    std::string line;
    std::string filename;
    std::vector<std::string> filenames;
    std::vector<std::string> parameters;
    int parm1, parm2;
    int count = 0;
    std::ifstream file("input.txt");
    if (!file) {
        char cwd[1024];
        if (getcwd(cwd, sizeof(cwd)) != NULL) {
            std::cout << "Current working directory: " << cwd << std::endl;
        }
        std::cout << "Unable to open file!" << std::endl;
        return 1;
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
        std::cout << param << std::endl;
        if (count == 0) {
            parm1 = std::stoi(param);
        } else if (count == 1) {
            parm2 = std::stoi(param);
        }
        count++;
    }
    int files_per_process = num_files / size;
    int start_file = rank * files_per_process;
    int end_file = (rank == size - 1) ? num_files : start_file + files_per_process;
    double start = omp_get_wtime();
    omp_set_num_threads(8);
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = start_file; i < end_file; ++i) {
        std::string fullPath = "./subsets/subset1" + filenames[i];
        std::ifstream ifile(fullPath);
        if (!ifile) {
            std::cout << "Subtable path error, can not open!" << std::endl;
            exit(1);
        }
        char cwd[1024];
        if (getcwd(cwd, sizeof(cwd)) != NULL) {
            std::cout << "Current working directory: " << cwd << std::endl;
        }
        Database db = TabularDatabase::fromFile(ifile, ',');
        CCFDMiner m(db, parm1, parm2);
        printf("I am process %d, thread %d, i=%d\n", rank, omp_get_thread_num(), i);
        m.run();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = omp_get_wtime();

    double total_time;
    MPI_Reduce(&end, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Total execution time: " << total_time - start << "s" << std::endl;
    }

    MPI_Finalize();
    return 0;
}

//int main() {
//    std::string line;
//    std::string filename;
//    std::vector<std::string> filenames;
//    std::vector<std::string> parameters;
//    int parm1, parm2;
//    int count = 0;
//    std::ifstream file("../input.txt");
//    if (!file) {
//        char cwd[1024];
//        if (getcwd(cwd, sizeof(cwd)) != NULL) {
//            std::cout << "Current working directory: " << cwd << std::endl;
//        }
//        std::cout << "Unable to open file!" << std::endl;
//        return 1;
//    }
//    std::getline(file, line);
//    std::stringstream ss(line);
//    while (ss >> filename) {
//        filenames.push_back(filename);
//    }
//    while (std::getline(file, line)) {
//        parameters.push_back(line);
//    }
//    for (const auto &param: parameters) {
//        std::cout << param << std::endl;
//        if (count == 0) {
//            parm1 = std::stoi(param);
//        } else if (count == 1) {
//            parm2 = std::stoi(param);
//        }
//        count++;
//    }
//        omp_set_num_threads(10);
//        double start = omp_get_wtime();
//#pragma omp parallel for schedule (dynamic, 1)
//        for (int i = 0; i < 7; ++i) {
//            std::string fullPath = "../subsets/" + filenames[i];
//            std::ifstream ifile(fullPath);
//            Database db = TabularDatabase::fromFile(ifile, ',');
//            CCFDMiner m(db, parm1, parm2);
//            printf("I am thread %d,i=%d\n", omp_get_thread_num(), i);
//            m.run();
//        }
//        double end = omp_get_wtime();
//        std::cout << "The total execution time is:" << float(end - start) << "s" << std::endl;
//        return 0;
//}

