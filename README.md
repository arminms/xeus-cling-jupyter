# xeus-cling-env
A bash script to build and install [xeus-cling](https://github.com/jupyter-xeus/xeus-cling) in a python virtual environment from source.

## Usage

```bash
$ ./make-xeus-cling-env.sh -h
Usage: ./make-xeus-cling-env.sh [OPTION]... [DIRECTORY(=~/xeus-cling(-env))]

Build and install xeus-cling 0.15.3 from source in DIRECTORY, with a python
virtual environment (optional) including jupyter, ipython, and ipykernel.
Needs CMake 3.12+ and 'realpath' to work (e.g. 'sudo apt-get install realpath').

  -b  FOLDER    build directory (default: ./build)
  -c            clean the build directory after installation
  -n  N         number of threads to build cling (default: 2)
  -h            show this help message
  -r            resume the build from the last step
  -s            skip creating python virtual environment
  -x            install extra libraries (xproperty, xwidgets, xtensor)
```
