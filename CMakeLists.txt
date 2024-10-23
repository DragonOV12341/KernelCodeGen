# preconfig
###################################################################
# set nvcc path
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.1/bin/nvcc)

# set to debug mode
# set(CMAKE_BUILD_TYPE Release)


# project config
###################################################################
cmake_minimum_required(VERSION 3.15.0)
project(KernelCodeGen LANGUAGES CXX C)    # delete CUDA

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

message(STATUS "LLVM_BUILD_LIBRARY_DIR ${LLVM_BUILD_LIBRARY_DIR}")
set(LLVM_BUILD_LIBRARY_DIR /home/pangyunfei/xushilong/KernelCodeGen/llvm-project/build/lib)
message(STATUS "LLVM_BUILD_LIBRARY_DIR ${LLVM_BUILD_LIBRARY_DIR}")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(MLIR REQUIRED CONFIG PATHS ./cmake)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
else()
  # Build via external projects mechanism
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(KernelCodeGen_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(KernelCodeGen_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${KernelCodeGen_SOURCE_DIR}/include)
include_directories(${KernelCodeGen_BINARY_DIR}/include)
include_directories(${KernelCodeGen_SOURCE_DIR}/build/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
add_compile_options(-g)


# optional config
###################################################################
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(config)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)

file(GLOB HEADERS_ROOT
  ${PROJECT_SOURCE_DIR}/include/*.h
  ${PROJECT_SOURCE_DIR}/include/*.hpp
)
file(GLOB HEADERS_SUBDIR_Frontend
  ${PROJECT_SOURCE_DIR}/include/Frontend/*.h
  ${PROJECT_SOURCE_DIR}/include/Frontend/*.hpp
)
file(GLOB HEADERS_SUBDIR_Backend
  ${PROJECT_SOURCE_DIR}/include/Backend/*.h
  ${PROJECT_SOURCE_DIR}/include/Backend/*.hpp
)
file(GLOB HEADERS_SUBDIR_IR
  ${PROJECT_SOURCE_DIR}/include/IR/*.h
  ${PROJECT_SOURCE_DIR}/include/IR/*.hpp
)

file(GLOB HEADERS_SUBDIR_Optimizer
  ${PROJECT_SOURCE_DIR}/include/Optimizer/*.h
  ${PROJECT_SOURCE_DIR}/include/Optimizer/*.hpp
)

foreach(header ${HEADERS_ROOT})
    install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/")
endforeach()
foreach(header ${HEADERS_SUBDIR_Frontend})
    install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Frontend")
endforeach()
foreach(header ${HEADERS_SUBDIR_Backend})
    install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Backend")
endforeach()
foreach(header ${HEADERS_SUBDIR_IR})
    install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/IR")
endforeach()
foreach(header ${HEADERS_SUBDIR_Optimizer})
    install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Optimizer")
endforeach()

install(
  TARGETS kcg_runtime
  EXPORT "kcg_runtime-targets"
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
