#define __HIP_PLATFORM_AMD__
#include <iostream>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <random>
#include <cmath>

#define MATRIX_SIZE 1024  // 矩阵的大小
#define CHECK_ROCBLAS_STATUS(status) \
    if (status != rocblas_status_success) { \
        std::cerr << "rocBLAS API call failed with error code: " << status << std::endl; \
        exit(EXIT_FAILURE); \
    }

void initialize_matrix(float* matrix, int rows, int cols) {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0,10);
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (distribution(generator));  // 生成随机半精度数值
    }
}

// 在 CPU 上执行 GEMM 计算（单精度）
void cpu_gemm(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

}

// 将半精度数组转换为单精度数组
void convert_half_to_float(const float* src, float* dst, int size) {
    for (int i = 0; i < size; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
}

int main() {
    rocblas_handle handle;
    CHECK_ROCBLAS_STATUS(rocblas_create_handle(&handle));

    const int N = MATRIX_SIZE;

    // 分配并初始化主机端内存
    float *h_A, *h_B, *h_C;
    h_A = new float[N * N];
    h_B = new float[N * N];
    h_C = new float[N * N];
    initialize_matrix(h_A, N, N);
    initialize_matrix(h_B, N, N);

    // 分配设备端内存
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, N * N * sizeof(float));
    hipMalloc(&d_B, N * N * sizeof(float));
    hipMalloc(&d_C, N * N * sizeof(float));

    // 将数据从主机复制到设备
    hipMemcpy(d_A, h_A, N * N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, N * N * sizeof(float), hipMemcpyHostToDevice);

    // 设置 alpha 和 beta
    float alpha{1};
    float beta{0};

    // 执行 GPU 矩阵乘法
    for(int i=0;i<5;++i){
        CHECK_ROCBLAS_STATUS(rocblas_sgemm(handle,
                                        rocblas_operation_none,
                                        rocblas_operation_none,
                                        N, N, N,
                                        &alpha,
                                        d_A, N,
                                        d_B, N,
                                        &beta,
                                        d_C, N));
    }

    // 将结果从设备端复制回主机端
    hipMemcpy(h_C, d_C, N * N * sizeof(float), hipMemcpyDeviceToHost);

    // 分配并计算 CPU 端的 GEMM 结果
    float *A_float = new float[N * N];
    float *B_float = new float[N * N];
    float *C_cpu_result = new float[N * N];
    convert_half_to_float(h_A, A_float, N * N);
    convert_half_to_float(h_B, B_float, N * N);
    cpu_gemm(A_float, B_float, C_cpu_result, N);

    // 验证 GPU 结果
    float max_error = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        float gpu_value = static_cast<float>(h_C[i]);
        float cpu_value = C_cpu_result[i];
        max_error = std::max(max_error, std::abs(gpu_value - cpu_value));
    }
    std::cout << "Max error between CPU and GPU results: " << max_error << std::endl;
    // print 10
    for(int i=0;i<10;++i){
        std::cout <<"cpu:" << C_cpu_result[i] << " gpu:" <<static_cast<float>(h_C[i]) << std::endl;
    }
    // 释放资源
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] A_float;
    delete[] B_float;
    delete[] C_cpu_result;
    rocblas_destroy_handle(handle);

    return 0;
}
