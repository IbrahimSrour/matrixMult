%%writefile matrixMult.cu


#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

__global__ void matMulKernel(const int *matA, const int *matB, int *matC, int matSize) {
  
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  matC[row * matSize + col] = 0;
  
  for (int k = 0; k < matSize; k++) {
    
    matC[row * matSize + col] += matA[row * matSize + k] * matB[k * matSize + col];
  }
}

int main() {
  int matSize = 4096;

  size_t byteSize = matSize * matSize * sizeof(int);

  vector<int> hostA(matSize * matSize);
  vector<int> hostB(matSize * matSize);
  vector<int> hostC(matSize * matSize);

  generate(hostA.begin(), hostA.end(), []() { return rand() % 100; });
  generate(hostB.begin(), hostB.end(), []() { return rand() % 100; });

  int *devA, *devB, *devC;
  cudaMalloc(&devA, byteSize);
  cudaMalloc(&devB, byteSize);
  cudaMalloc(&devC, byteSize);
cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  cudaMemcpy(devA, hostA.data(), byteSize, cudaMemcpyHostToDevice);
  cudaMemcpy(devB, hostB.data(), byteSize, cudaMemcpyHostToDevice);

  int THREADS = 32;
  int BLOCKS = matSize / THREADS;

  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  matMulKernel<<<blocks, threads>>>(devA, devB, devC, matSize);

  cudaMemcpy(hostC.data(), devC, byteSize, cudaMemcpyDeviceToHost);
cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  std::cout << "Elapsed time: " << elapsedTime << " ms\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);

  return 0;
}
