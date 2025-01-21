# xeus-cling-jupyter
A bash script to build and install [xeus-cling](https://github.com/jupyter-xeus/xeus-cling) from source in a python virtual environment or as a Docker image.

## Usage

```bash
$ git clone git@github.com:arminms/xeus-cling-jupyter.git
$ cd xeus-cling-jupyter
$ ./make-xeus-cling-jupyter.sh --help
Usage: ./make-xeus-cling-jupyter.sh [OPTION]... [DIRECTORY(=~/xeus-cling(-env))]

Build and install xeus-cling 0.15.3 from source in DIRECTORY, optionally with a
python virtual environment or as a Docker image.
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

## Running using Docker
Use the following command to launch the [Docker](https://docker.io) container:
```
docker run -p 8888:8888 -it --rm asobhani/xeus-cling-jupyter
```
Or this one with CUDA support:
```
docker run --gpus=all -p 8888:8888 -it --rm asobhani/xeus-cling-jupyter:latest-cuda
```
