PROJECT(SCFDM_all CXX)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FIND_PACKAGE( Boost 1.40 COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
find_package(LLVM REQUIRED CONFIG)
find_package(MPI REQUIRED)

set(SOURCE_FILES main.cpp)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall -O2 -Wno-sign-compare -Wextra -lstdc++")
FILE(GLOB_RECURSE DataSrcs data/*.cpp)
FILE(GLOB_RECURSE AlgSrcs algorithms/*.cpp)
FILE(GLOB_RECURSE UtilSrcs util/*.cpp)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(${LLVM_INCLUDE_DIRS} )
add_definitions(${LLVM_DEFINITIONS})
ADD_EXECUTABLE(SCFDM_all main.cpp ${DataSrcs} ${AlgSrcs} ${UtilSrcs})

# 调试配置
set(CMAKE_BUILD_TYPE "Debug")

# openMP 配置
FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

llvm_map_components_to_libnames(llvm_libs support core irreader)

target_link_libraries(SCFDM_all PRIVATE MPI::MPI_CXX)