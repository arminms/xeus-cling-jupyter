ARG CUDA=0

#-- base image -----------------------------------------------------------------

FROM ubuntu:20.04 AS base

# change default shell to bash
SHELL ["/bin/bash", "-c"]

# install python3
RUN set -ex \
    && apt-get update && apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 \
    && rm -rf /var/lib/apt/lists/*

#-- base-0 image (no CUDA) -----------------------------------------------------

FROM base AS base-0

# change default shell to bash
SHELL ["/bin/bash", "-c"]

#-- base-10 image (CUDA 10) -----------------------------------------------------

FROM base AS base-10

# change default shell to bash
SHELL ["/bin/bash", "-c"]

# install nvidia-cuda-toolkit 10
RUN set -ex \
    && apt-get update && apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        nvtop \
        nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

#-- base-11 image (CUDA 11) -----------------------------------------------------

FROM nvidia/cuda:11.8.0-devel-ubuntu20.04 AS base-11

# change default shell to bash
SHELL ["/bin/bash", "-c"]

# install python3
RUN set -ex \
    && apt-get update && apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        nvtop \
        python3 \
    && rm -rf /var/lib/apt/lists/*

#-- base-12 image (CUDA 12) -----------------------------------------------------

FROM nvidia/cuda:12.6.0-devel-ubuntu20.04 AS base-12

# change default shell to bash
SHELL ["/bin/bash", "-c"]

# install python3
RUN set -ex \
    && apt-get update && apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        nvtop \
        python3 \
    && rm -rf /var/lib/apt/lists/*

#-- jupyter image --------------------------------------------------------------

FROM base AS jupyter

# change default shell to bash
SHELL ["/bin/bash", "-c"]

# install virtualenv
RUN set -ex \
    && apt-get update && apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        virtualenv \
    && rm -rf /var/lib/apt/lists/*

# create a venv with jupyter
RUN set -ex \
    && virtualenv /opt/xeus-cling \
    && source /opt/xeus-cling/bin/activate \
    && pip install --upgrade pip \
    && pip install jupyter ipython ipykernel \
    && deactivate

#-- xeus-cling image -----------------------------------------------------------

FROM base AS xeus-cling

# change default shell to bash
SHELL ["/bin/bash", "-c"]

# install build-essential and other dependencies
RUN set -ex \
    && apt-get update && apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        gnutls-dev \
        libssl-dev \
        nvidia-cuda-toolkit \
        pkg-config \
        uuid-dev \
    && rm -rf /var/lib/apt/lists/*

# install xeus-cling
RUN cd /opt \
    && git clone https://github.com/arminms/xeus-cling-env.git \
    && cd /opt/xeus-cling-env \
    && ./make-xeus-cling-env.sh -rsxn 2 /opt/xeus-cling

#-- xeus-cling-jupyter image ---------------------------------------------------

FROM base-${CUDA} AS xeus-cling-jupyter

# reintroduce the CUDA argument
ARG CUDA

# set the maintainer and description
LABEL maintainer="Armin Sobhani <arminms@gmail.com>"
LABEL description="A Jupyter image with xeus-cling and optionally CUDA support"

# change default shell to bash
SHELL ["/bin/bash", "-c"]

# install python, adduser and other dependencies
RUN set -ex \
    && apt-get update && apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        adduser \
        ca-certificates \
        git \
        libomp-9-dev \
        libstdc++-9-dev \
        libtbb-dev \
        wget \
    && rm -rf /var/lib/apt/lists/*

# copy the venv and xeus-cling
COPY --from=jupyter /opt/xeus-cling /opt/xeus-cling
COPY --from=xeus-cling /opt/xeus-cling /opt/xeus-cling

# remove the CUDA kernels if CUDA is not installed
RUN if [ "$CUDA" = "0" ] ; then rm -rf /opt/xeus-cling/share/jupyter/kernels/*-cuda ; fi

# create a non-root user
RUN set -ex && \
    adduser --disabled-password --gecos "Default Jupyter user" \
    --uid 1000 \
    jovyan

# Run as the non-root user we just created
USER 1000

# expose the jupyter-lab port
EXPOSE 8888

# set the working directory
WORKDIR /home/jovyan

# activate the venv and start jupyter-lab
CMD source /opt/xeus-cling/bin/activate ; SHELL=/bin/bash jupyter-lab "--no-browser"
