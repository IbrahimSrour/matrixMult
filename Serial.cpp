#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <chrono>

using std::cout;
using std::generate;
using std::vector;

void matMul(const vector<int>& matA, const vector<int>& matB, vector<int>& matC, int matSize) {
  for(int row = 0; row < matSize; row++) {
    for(int col = 0; col < matSize; col++) {
      matC[row * matSize + col] = 0;
      for(int k = 0; k < matSize; k++) {
        matC[row * matSize + col] += matA[row * matSize + k] * matB[k * matSize + col];
      }
    }
  }
}

int main() {
  int matSize = 2048;

  vector<int> hostA(matSize * matSize);
  vector<int> hostB(matSize * matSize);
  vector<int> hostC(matSize * matSize);

  generate(hostA.begin(), hostA.end(), []() { return rand() % 100; });
  generate(hostB.begin(), hostB.end(), []() { return rand() % 100; });

  auto start = std::chrono::high_resolution_clock::now();
  matMul(hostA, hostB, hostC, matSize);
  auto stop = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

  std::cout << "Elapsed time: " << duration.count() << " ms\n";

  return 0;
}
