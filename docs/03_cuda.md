---
subject: Xeus-Cling Quickstart Tutorial
kernelspec:
  name: xcpp17
  display_name: C++17
  language: C++17
---

# Working with CUDA

```{code-cell} cpp
:tags: [skip-execution]
#include <cuda.h>
#include <cuda_runtime.h>
```
+++
```{code-cell} cpp
:tags: [skip-execution]
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
prop.name
```
+++
```{code-cell} cpp
void vector_add(float *out, float *a, float *b, int n)
{    for(int i = 0; i < n; i++)
        out[i] = a[i] + b[i];
}
```
+++
```{code-cell} cpp

const size_t N{10'000'007};

float *a, *b, *out; 

// Allocate memory
a   = (float*)malloc(sizeof(float) * N);
b   = (float*)malloc(sizeof(float) * N);
out = (float*)malloc(sizeof(float) * N);

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
:tags: [skip-execution]

__global__ void cuda_vector_add(float *out, float *a, float *b, int n)
{    for(int i = 0; i < n; i++)
        out[i] = a[i] + b[i];
}
```
+++
```{code-cell} cpp
:tags: [skip-execution]

float *d_a, *d_b, *d_out;

a = (float*)malloc(sizeof(float) * N);

// Allocate device memory for a
cudaMalloc((void**)&d_a,   sizeof(float) * N);
cudaMalloc((void**)&d_b,   sizeof(float) * N);
cudaMalloc((void**)&d_out, sizeof(float) * N);

// Transfer data from host to device memory
cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
```
+++
```{code-cell} cpp
:tags: [skip-execution]

%%timeit -n 3 -r 3
cuda_vector_add<<<1,1>>>(d_out, d_a, d_b, N);
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
