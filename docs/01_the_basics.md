---
subject: Xeus-Cling Quickstart Tutorial
kernelspec:
  name: xcpp17
  display_name: C++17
  language: C++17
---

# The Basics

## Declaration shadowing

With `cling` you can redefine a function, variable, or class whose definition was already provided for a particular interpreter session. That's a big quality-of-life improvement for interpreted <wiki:C++> because _without_ it, for instance, if you start with this in a code cell:

```{code} cpp
double foo(double a)
{
    return a * a;
}
```

And later want to change `foo()` to the following, you'll get redefinition error:

```{code} cpp
double foo(double a)
{
    return a * a * a;
}
```

Support for redefinitions is automatically enabled for Jupyter notebooks. You can manually turn it `on`/`off` as follows:

```{code} cpp
gClingOpts->AllowRedefinition = 1; // or 0 to disable
```

:::{seealso}
If you're interested to know how this feature is implemented check <doi:10.1145/3377555.3377901>.
:::



## Using third-party libraries

When building a binary, you usually specify the include directories and the library path of third-party libraries in the build tool. The library will be loaded upon binary execution.

xeus-cling is slightly different, it allows you to specify both include directories and library path, however you need to load the library explicitly. This is done with special pragma commands that you can use in a code cell in a Jupyter Notebook:

```{code} cpp
#pragma cling add_include_path("include/directory")
#pragma cling add_library_path("lib/directory")
#pragma cling load("libname")
```

## Magic commands

Magics are special commands for the kernel that are not part of the C++ programming language.

They are defined with the symbol `%` for a line magic and `%%` for a cell magic.

### `%%executable`
Dump the code from all entered cells into an executable binary. The content of the cell is used for the body of the main function.

```
%%executable filename [-- linker options]
```
_Example:_
```{code-cell} cpp
#include <iostream>
```
+++
```{code-cell} cpp
int square(int x) { return x * x; }
```
+++
```{code-cell} cpp
:tags: [hide-output]

%%executable square.x
std::cout << square(4) << std::endl;
```
+++
```{code-cell} cpp
:tags: [hide-output]
!./square.x
```

### `%%file`

This magic command copies the content of the cell in a file named `filename`. There's an optional `-a` argument to append the content to the file.

```
%%file [-a] filename
```

_Example:_
```{code-cell} cpp
%%file tmp.txt
Demo of magic command
```
+++
```{code-cell} cpp
%%file -a tmp.txt
Appending to tmp.txt
```
+++
```{code-cell} cpp
:tags: [hide-output]
!cat tmp.txt
```
### `%timeit`

Measures the execution time for a line statement (`%timeit`) or for a block of statements (`%%timeit`).

_Usage in line mode:_
```
%timeit [-n<N> -r<R> -p<P>] statement
```
_Usage in cell mode:_
```
%%timeit [-n<N> -r<R> -p<P>]
statements
```
_Example_:
```{code-cell} cpp
#include <xtensor/xtensor.hpp>
```
+++
```{code-cell} cpp
auto x = xt::linspace<double>(1, 10, 100);
```
+++
```{code-cell} cpp
:tags: [hide-output]
%timeit xt::eval(xt::sin(x));
```
+++
```{code-cell} cpp
:tags: [hide-output]
%timeit -n 10 -r 1 -p 6 xt::eval(xt::sin(x));
```
+++
```{code-cell} cpp
:tags: [hide-output]
%%timeit 
auto y = xt::linspace<double>(1, 10, 100);
xt::eval(xt::sin(y) * xt::cos(x));
```
_Optional arguments:_
|||
|-|-|
|`-n`|execute the given statement <N> times in a loop. If this value is not given, a fitting value is chosen.|
|`-r`|repeat the loop iteration <R> times and take the best result. Default: 7|
|`-p`|use a precision of <P> digits to display the timing result. Default: 3|