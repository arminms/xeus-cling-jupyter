---
subject: Xeus-Cling Quickstart Tutorial
kernelspec:
  name: xcpp14
  display_name: C++14
  language: C++14
---

# Working with CUDA

Here's the same vector addition example the we used for <wiki:OpenMP> with <wiki:CUDA>.

:::{tip} Choosing the right kernel 
Make sure the selected kernel for the notebook is `C++14-CUDA`.
:::

```{code-cell} cpp
:tags: [skip-execution]
#include <iostream>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>
```
+++
```{code-cell} cpp
:tags: [skip-execution]
cudaDeviceProp devProp;
cudaError_t cudaStatus = cudaGetDeviceProperties(&devProp, 0);
if (cudaStatus == cudaSuccess)
    std::cout << "  Name\t\t\t\t: " << devProp.name << '\n'
              << "  Total global memory\t\t: " << devProp.totalGlobalMem << " bytes\n"
              << "  Total shared memory per block\t: " << devProp.sharedMemPerBlock << " bytes\n"
              << "  Total registers per block\t: " << devProp.regsPerBlock << '\n'
              << "  Warp size\t\t\t: " << devProp.warpSize << '\n'
              << "  Maximum threads per block\t: " << devProp.maxThreadsPerBlock << '\n'
              << "  Number of multiprocessors\t: " << devProp.multiProcessorCount << '\n'
              << "  CUDA Capability\t\t: " << devProp.major << '.' << devProp.minor << std::endl;
else
    std::cerr << "cudaGetDeviceProperties failed: " << cudaGetErrorString(cudaStatus) << std::endl;

int runtimeVersion;
cudaStatus = cudaRuntimeGetVersion(&runtimeVersion);
if (cudaStatus == cudaSuccess)
{
    int major = runtimeVersion / 1000;
    int minor = (runtimeVersion % 1000) / 10;
    int patch = runtimeVersion % 10;
    std::cout << "  CUDA Runtime Version\t\t: " << major << '.' << minor << '.' << patch << std::endl;
}
else
    std::cerr << "Error getting CUDA Runtime Version: " << cudaGetErrorString(cudaStatus) << std::endl;
```
+++
```{code-cell} cpp
template <typename T>
void vector_add(T* out, T* a, T *b, size_t n)
{    for(int i = 0; i < n; i++)
        out[i] = a[i] + b[i];
}
```
+++
```{code-cell} cpp

const size_t N{100'000'007};

double *a, *b, *out; 

// Allocate memory
a   = (double*)malloc(sizeof(double) * N);
b   = (double*)malloc(sizeof(double) * N);
out = (double*)malloc(sizeof(double) * N);

// Initialize array
for(int i = 0; i < N; i++)
{    a[i] = 1.0f; b[i] = 2.0f;
}
```
+++
```{code-cell} cpp
:tags: [hide-output]
%%timeit
vector_add(out, a, b, N);
```
+++
```{code-cell} cpp
:tags: [hide-output]
out[1]
```
+++
```{code-cell} cpp
:tags: [skip-execution]

template <typename T>
__global__ void cuda_vector_add(T *out, T *a, T *b, size_t n)
{   auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < n) out[idx] = a[idx] + b[idx];
}
```
+++
```{code-cell} cpp
:tags: [skip-execution]

double *d_a, *d_b, *d_out;

a = (double*)malloc(sizeof(double) * N);

// Allocate device memory for a
cudaMalloc((void**)&d_a,   sizeof(double) * N);
cudaMalloc((void**)&d_b,   sizeof(double) * N);
cudaMalloc((void**)&d_out, sizeof(double) * N);

// Transfer data from host to device memory
cudaMemcpy(d_a, a, sizeof(double) * N, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, b, sizeof(double) * N, cudaMemcpyHostToDevice);
```
+++
```{code-cell} cpp
:tags: [skip-execution]

const size_t threads_per_block{256};
size_t blocks_in_grid{N / threads_per_block + 1};
```
+++
```{code-cell} cpp
:tags: [skip-execution]

%%timeit
cuda_vector_add<<<blocks_in_grid, threads_per_block>>>(d_out, d_a, d_b, N);
cudaDeviceSynchronize();
```
+++
```{code-cell} cpp
:tags: [skip-execution]

// Transfer data from device to host memory
cudaMemcpy(d_out, out, sizeof(double) * N, cudaMemcpyDeviceToHost);
out[1]
```
+++
```{code-cell} cpp
:tags: [skip-execution]

// Cleanup after kernel execution
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_out);
```
+++
```{code-cell} cpp
free(a);
free(b);
free(out);
```
