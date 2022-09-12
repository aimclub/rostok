FROM ubuntu:22.04

RUN apt-get update 

RUN apt-get install -y x11-apps
RUN apt-get install -y ffmpeg libsm6 libxext6 


# Install base utilities
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH


RUN conda config --add channels https://conda.anaconda.org/conda-forge
RUN conda config --add channels https://conda.anaconda.org/intel 

RUN conda install -c conda-forge numpy irrlicht scipy
RUN conda install -c projectchrono pychrono=7.0.0

ENV DISPLAY=host.docker.internal:0.0