FROM ubuntu:20.04 AS builder

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
        pkg-config \
        libssl-dev \
        nvidia-cuda-toolkit \
        python3 \
        uuid-dev \
    && rm -rf /var/lib/apt/lists/*

# install xeus-cling
RUN cd /opt \
    && git clone https://github.com/arminms/xeus-cling-env.git \
    && cd /opt/xeus-cling-env \
    && ./make-xeus-cling-env.sh -srxn 2 /opt/xeus-cling

FROM ubuntu:20.04 AS runtime

# change default shell to bash
SHELL ["/bin/bash", "-c"]

# install python, adduser and other dependencies
RUN set -ex \
    && apt-get update && apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        adduser \
        libomp-9-dev \
        libstdc++-9-dev \
        libtbb-dev \
        nvidia-cuda-toolkit \
        python3 \
        virtualenv \
        wget \
    && rm -rf /var/lib/apt/lists/*

# create a non-root user
RUN set -ex && \
    adduser --disabled-password --gecos "Default Jupyter user" \
    --uid 1000 \
    jovyan

# create a venv to install packages into
RUN set -ex \
    && virtualenv /opt/xeus-cling \
    && source /opt/xeus-cling/bin/activate \
    && pip install --upgrade pip \
    && pip install jupyter ipython ipykernel \
    && deactivate

COPY --from=builder /opt/xeus-cling /opt/xeus-cling

# Run as the non-root user we just created
USER 1000

# expose the jupyter-lab port
EXPOSE 8888

# activate the venv and start jupyter-lab
CMD cd ~ ; source /opt/xeus-cling/bin/activate ; SHELL=/bin/bash jupyter-lab "--no-browser"
