FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -qq -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3-dev \
    libglu1-mesa-dev \
    libgtk2.0-0 \
    libhdf5-dev \
    libjpeg-dev \
    liblmdb-dev \
    libopenblas-dev \
    libosmesa6-dev \
    libpng-dev \
    libsm6 \
    libxcursor-dev \
    libxext6 \
    libxi-dev \
    libxinerama-dev \
    libxrandr-dev \
    lxde \
    mesa-utils \
    ninja-build \
    pkg-config \
    unzip \
    vim \
    wget && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

RUN /opt/conda/bin/conda install -c anaconda protobuf
RUN /opt/conda/bin/pip install --upgrade pip
RUN /opt/conda/bin/pip install --upgrade --ignore-installed setuptools
RUN /opt/conda/bin/pip install \
    addict \
    fire \
    future \
    h5py \
    imageio \
    imgaug \
    joblib \
    lmdb \
    matplotlib \
    ninja \
    numpy \
    open3d-python==0.7.0.0 \
    opencv-python \
    pandas \
    Pillow \
    protobuf \
    pyflann3==1.8.4.1 \
    pytorch-ignite==0.2.0 \
    pytz \
    PyYAML \
    scikit-image \
    scikit-learn \
    scipy \
    tensorboard \
    torch \
    torchvision \
    tqdm \
    typing
RUN /opt/conda/bin/pip install git+https://github.com/rbgirshick/yacs
RUN /opt/conda/bin/pip install git+https://github.com/Cadene/pretrained-models.pytorch
# RUN /opt/conda/bin/conda install -c conda-forge pyflann=1.8.4 
RUN /opt/conda/bin/conda clean -ya

ENV LD_LIBRARY_PATH .:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CPATH /usr/local/cuda/include:$CPATH
ENV PATH /usr/local/cuda/bin:/opt/conda/bin:$PATH
