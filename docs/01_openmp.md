---
subject: Xeus-Cling Quickstart Tutorial
kernelspec:
  name: xcpp17-openmp
  display_name: C++17
  language: C++17-OpenMP
---

# Working with OpenMP

```{code-cell} cpp
#include <vector>

#pragma cling load("libomp.so.5")
```
+++
```{code-cell} cpp
template <typename OutputIt, typename InputIt, typename Size>
void vector_add(OutputIt out, InputIt a, InputIt b, Size n)
{
    for (Size i = 0; i < n; ++i)
        out[i] = a[i] + b[i];
}
```
+++
```{code-cell} cpp
template <typename OutputIt, typename InputIt, typename Size>
void vector_add_omp(OutputIt out, InputIt a, InputIt b, Size n)
{
    #pragma omp parallel for
    for(Size i = 0; i < n; ++i)
        out[i] = a[i] + b[i];
}
```
+++
```{code-cell} cpp
const size_t N{10'000'000};

std::vector<float> a(N, 1.0f), b(N, 2.0f), out(N);
```
+++
```{code-cell} cpp
:tags: [hide-output]
%%timeit
vector_add(out.begin(), a.cbegin(), b.cbegin(), N);
```
+++
```{code-cell} cpp
:tags: [hide-output]
out[0]
```
+++
```{code-cell} cpp
:tags: [hide-output]
%%timeit
vector_add_omp(out.begin(), a.cbegin(), b.cbegin(), N);
```
+++
```{code-cell} cpp
:tags: [hide-output]
out[0]
```
+++
