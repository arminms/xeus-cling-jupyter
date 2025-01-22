---
title: Jupyter with Xeus-Cling Kernel
subtitle: Container image and Python virtual environment
description: Getting started using xeus-cling
---

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arminms/xeus-cling-jupyter/HEAD)

:::::{aside}
::::{important} Try with Docker
:class: dropdown

Use the following command to launch the [Jupyter](wiki:Project_Jupyter) + [Xeus-Cling](xref:xeus-cling) with [Docker](wiki:Docker_(software)):
```
docker run -p 8888:8888 -it --rm asobhani/xeus-cling-jupyter
```
Or this one with CUDA support:
```
docker run --gpus=all -p 8888:8888 -it --rm asobhani/xeus-cling-jupyter:latest-cuda
```
::::
:::::

This guide helps you build and install the entire software stack for a [Jupyter](wiki:Project_Jupyter) with [Xeus-Cling](xref:xeus-cling) kernel from the source either as a [Docker](wiki:Docker_(software)) container image or in a [Python](wiki:Python_(programming_language)) virtual environment.