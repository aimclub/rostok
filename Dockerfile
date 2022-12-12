FROM ubuntu:22.04

RUN apt-get update 

RUN apt-get install -y x11-apps
RUN apt-get install -y ffmpeg libsm6 libxext6 


# Install base utilities
RUN apt-get update && \
    apt-get install -y wget git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
COPY environment_copy.yml .
RUN conda env create -f environment.yml
ENV DISPLAY=host.docker.internal:0.0