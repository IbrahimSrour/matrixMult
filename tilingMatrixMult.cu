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

const int MATRIX_SIZE = 2048;
const int SHARED_MEM_SIZE = 1024;

__global__ void matMul(const int *mat_a, const int *mat_b, int *mat_c) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int shared_a[SHARED_MEM_SIZE];
  __shared__ int shared_b[SHARED_MEM_SIZE];

  int temp = 0;

  for (int i = 0; i < MATRIX_SIZE; i += blockDim.x) {
    shared_a[threadIdx.y * blockDim.x + threadIdx.x] = mat_a[row * MATRIX_SIZE + i + threadIdx.x];
    shared_b[threadIdx.y * blockDim.x + threadIdx.x] = mat_b[i * MATRIX_SIZE + threadIdx.y * MATRIX_SIZE + col];

    __syncthreads();

    for (int j = 0; j < blockDim.x; j++) {
      temp += shared_a[threadIdx.y * blockDim.x + j] * shared_b[j * blockDim.x + threadIdx.x];
    }

    __syncthreads();
  }

  mat_c[row * MATRIX_SIZE + col] = temp;
}


int main() {
  
  size_t bytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(int);

  vector<int> h_mat_a(MATRIX_SIZE * MATRIX_SIZE);
  vector<int> h_mat_b(MATRIX_SIZE * MATRIX_SIZE);
  vector<int> h_mat_c(MATRIX_SIZE * MATRIX_SIZE);

  generate(h_mat_a.begin(), h_mat_a.end(), []() { return rand() % 100; });
  generate(h_mat_b.begin(), h_mat_b.end(), []() { return rand() % 100; });

  int *d_mat_a, *d_mat_b, *d_mat_c;
  cudaMalloc(&d_mat_a, bytes);
  cudaMalloc(&d_mat_b, bytes);
  cudaMalloc(&d_mat_c, bytes);
 cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, 0);

cudaMemcpy(d_mat_a, h_mat_a.data(), bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_mat_b, h_mat_b.data(), bytes, cudaMemcpyHostToDevice);

int THREADS = 32;
int BLOCKS = MATRIX_SIZE / THREADS;

dim3 threads(THREADS, THREADS);
dim3 blocks(BLOCKS, BLOCKS);

matMul<<<blocks, threads>>>(d_mat_a, d_mat_b, d_mat_c);

cudaMemcpy(h_mat_c.data(), d_mat_c, bytes, cudaMemcpyDeviceToHost);

cudaEventRecord(stop, 0);  // Move this line up here.
cudaEventSynchronize(stop);

float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);

std::cout << "Elapsed time: " << elapsedTime << " ms\n";

cudaEventDestroy(start);
cudaEventDestroy(stop);

cudaFree(d_mat_a);
cudaFree(d_mat_b);
cudaFree(d_mat_c);

return 0;
}
