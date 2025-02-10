---
subject: Xeus-Cling Quickstart Tutorial
kernelspec:
  name: xcpp17
  display_name: C++17
  language: C++17
---

# C++17 Parallel Algorithms

Here's a simple example of using [`C++17` (parallel) execution policies](https://en.cppreference.com/w/cpp/algorithm#Execution_policies) for <wiki:summation>.

:::{tip} Choosing the right kernel 
Make sure the selected kernel for the notebook is `C++17`.
:::

```{code-cell} cpp
#include <vector>
#include <execution>
```
+++
We have to load <wiki:Threading_Building_Blocks> library that under the hood does the actual parallelization: 

```{code-cell} cpp
#pragma cling load("libtbb.so.2")
```
+++
```{code-cell} cpp
const std::vector<double> v(10'000'007, 0.1);
```
+++
```{code-cell} cpp
:tags: [hide-output]

%%timeit
std::reduce(std::execution::seq, v.cbegin(), v.cend());
```
+++
```{code-cell} cpp
:tags: [hide-output]

%%timeit
std::reduce(std::execution::par, v.cbegin(), v.cend());
```
+++
+++
```{code-cell} cpp
:tags: [hide-output]

auto s = std::reduce(std::execution::par, v.cbegin(), v.cend());
s
```
