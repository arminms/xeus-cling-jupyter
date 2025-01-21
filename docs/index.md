---
title: Jupyter with Xeus-Cling Kernel
subtitle: Python virtual environment and the Dockerfile
description: Getting started using xeus-cling
---

:::::{aside}
::::{important} Try with Docker
:class: dropdown

Use the following command to launch the [Jupyter](wiki:Project_Jupyter) + [xeus-cling](xref:xeus-cling) with [Docker](wiki:Docker_(software)):
```
docker run -p 8888:8888 -it --rm asobhani/xeus-cling-jupyter
```
Or this one with CUDA support:
```
docker run --gpus=all -p 8888:8888 -it --rm asobhani/xeus-cling-jupyter:latest-cuda
```
::::
:::::

This guide helps you build and install the entire software stack for a [Jupyter](wiki:Project_Jupyter) with [xeus-cling](xref:xeus-cling) kernel from the source either in a [Python](wiki:Python_(programming_language)) virtual environment or as a [Docker](wiki:Docker_(software)) container image.