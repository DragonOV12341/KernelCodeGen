cmake_minimum_required(VERSION 3.22)
project(myRocGEMM)
link_directories(
    "/opt/dtk/rocblas/lib"
    "/opt/dtk/hip/lib/"
    )
add_compile_options(-g)
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} rocblas amdhip64)
