---
title: Start using Xeus-Cling
subject: Xeus-Cling Quickstart
subtitle: Building and installation
short_title: Get Started
description: Running xeus-cling in a virtual environment or a Docker container.
---

# Start using Xeus-Cling

---

(requirements)=
## Requirements

- <wiki:Linux> OS (tested on <wiki:Ubuntu> 20.04).
- <wiki:Linux> build essentials (e.g. `sudo apt-get installbuild-essential` on <wiki:Ubuntu>).
- 'realpath' (e.g. `sudo apt-get install realpath` on <wiki:Ubuntu>).
- <wiki:CMake> 3.12 or higher.
- <wiki:CUDA> CUDA >= 8.0 and <= 10.2 if you need CUDA kernels (`-g` option).
- [Docker](wiki:Docker_(software)) if you want to build/run the `Docker` images/containers.

(build_virtual_env)=
## Build the entire software stack

To build and install the entire software stack for running [Jupyter](wiki:Project_Jupyter) with [Xeus-Cling](xref:xeus-cling) kernel from the source, you can use [](https://github.com/arminms/xeus-cling-jupyter/blob/main/make-xeus-cling-jupyter.sh#L24-L46) bash script:

``` shell
$ git clone git@github.com:arminms/xeus-cling-jupyter.git
$ cd xeus-cling-jupyter
$ ./make-xeus-cling-jupyter.sh --help
```

:::{dropdown} Output
:open:

```
Usage: ./make-xeus-cling-jupyter.sh [OPTION]... [DIRECTORY(=~/xeus-cling(-env))]

Build and install xeus-cling 0.15.3 from source in DIRECTORY, optionally with a python virtual environment or as a Docker image.
Needs CMake 3.12+ and 'realpath' to work (e.g. 'sudo apt-get install realpath').

  -b  FOLDER    build directory (default: ./build)
  -c            clean the build directory after installation
  -d            build xeus-cling-jupyter Docker image (requires Docker)
  -g            install xeus-cling CUDA-enabled kernels for NVIDIA GPUs
  -k            install kernels to the current user's kernel registry
  -n  N         number of threads to build cling (default: 2)
  -h            show this help message
  -r            resume the build from the last step
  -s            skip creating python virtual environment
  -x            install extra libraries (xproperty, xwidgets, xtensor)
```
:::

For instance, to build and install in a virtual environment named `xeus-cling-env` in your home directory (`~`) with extra libraries (`-x`) using 4 concurrent threads, you can use the following command:

``` shell
./make-xeus-cling-jupyter.sh -xn 4 ~/xeus-cling-env
```

::::{attention} Be patient ⏲️
:class: dropdown
:open:

Building the the entire software stack for [xeus-cling](xref:xeus-cling), specially the `cling` part, may take a long time even with several threads. A good time to enjoy a cup of coffee or tea! ☕🍵
::::

::::{tip} Memory footprint 💻
:class: dropdown
:open:

Building `cling` requires a lot of RAM. Be careful when setting the number of threads. Hyperthreading can drastically impact the memory usage. Here are some recommended combinations:
- 4 Threads with 32 GB RAM
- 12 Threads with 128 GB RAM
::::

(build_docker_image)=
## Building Docker images

To build the image using [Docker](wiki:Docker_(software)) simply run the script with the `-d` option:
```bash
./make-xeus-cling-jupyter.sh -d
```
Or `-gd` for CUDA support:
```bash
./make-xeus-cling-jupyter.sh -gd
```

(launch_docker_image)=
## Launching pre-built container images

Use the following command to launch the the pre-build container image:
```bash
docker run -p 8888:8888 -it --rm asobhani/xeus-cling-jupyter
```
Or this one with CUDA support:
```bash
docker run --gpus=all -p 8888:8888 -it --rm asobhani/xeus-cling-jupyter:latest-cuda
```

You can also use the above container images as the starting point for your custom-made docker image (e.g. `FROM asobhani/xeus-cling-jupyter:latest`).