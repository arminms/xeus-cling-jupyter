---
title: Jupyter with Xeus-Cling Kernel
subtitle: Python virtual environment and container image
description: Getting started using Xeus-Cling
---

[![GitHub License](https://img.shields.io/github/license/arminms/xeus-cling-env?logo=github&logoColor=lightgrey&color=green)](https://github.com/arminms/xeus-cling-env/blob/main/LICENSE)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arminms/xeus-cling-jupyter/HEAD)


:::::{aside}

::::{important} Try in a Container
:class: dropdown
[Docker:](wiki:Docker_(software))
```bash
docker run -p 8888:8888 -it --rm asobhani/xeus-cling-jupyter
```
Or this one with <wiki:CUDA> support:
```bash
docker run --gpus=all -p 8888:8888 -it --rm asobhani/xeus-cling-jupyter:latest-cuda
```
[Apptainer:](wiki:Singularity_(software))
```bash
apptainer run docker://asobhani/xeus-cling-jupyter:latest
```
Or this one with <wiki:CUDA> support:
```bash
apptainer run --nv docker://asobhani/xeus-cling-jupyter:latest-cuda
```
::::

:::::{seealso} Try on Binder
:class: dropdown
::::{grid} 2 2 2 2
:::{grid-item}
👉   [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arminms/xeus-cling-jupyter/HEAD)
:::
:::{grid-item}
_Be advised sometimes it takes several minutes to start!_
:::
::::

:::::

This guide helps you build and install the entire software stack for [Jupyter](wiki:Project_Jupyter) with [Xeus-Cling](xref:xeus-cling) kernel from the source either in a [Python](wiki:Python_(programming_language)) virtual environment or as a [Docker](wiki:Docker_(software)) container image.