#include <iostream>
#include <cuda_runtime.h>

__global__ void vecAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 10;
    float a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float b[10] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    float c[10];
    
    float *dev_a, *dev_b, *dev_c;
    
    cudaMalloc(&dev_a, n * sizeof(float));
    cudaMalloc(&dev_b, n * sizeof(float)); 
    cudaMalloc(&dev_c, n * sizeof(float));
    
    cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    
    int block_size = 32;
    int grid_size = (n + block_size - 1) / block_size;
    vecAdd<<<grid_size, block_size>>>(dev_a, dev_b, dev_c, n);
    
    cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "First few results: ";
    for(int i = 0; i < 5; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    return 0;
}
