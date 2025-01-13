FROM ubuntu:22.04

RUN apt-get update 

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
RUN useradd -m jovyan

ENV PATH=$CONDA_DIR/bin:$PATH
COPY environment_jmoves.yml .

# RUN conda env create -f environment_jmoves.yml
RUN conda env update -n base --file environment_jmoves.yml
RUN conda init

SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

# Install jupiter stuff
RUN conda run -n base pip install jupyter notebook voila
RUN conda run -n base jupyter server extension enable voila --sys-prefix --enable_nbextensions=True

# Install our package 
WORKDIR /home/jovyan

COPY . ./jmoves_env
RUN conda run -n base pip install -e ./jmoves_env
RUN conda run -n base pip install ./jmoves_env/meshcat-0.3.2.tar.gz


USER jovyan
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda init

CMD ["/bin/bash", "-c", "if [ -d /usr/local/bin/before-notebook.d ]; then for file in /usr/local/bin/before-notebook.d/*; do $file ; done; fi && jupyter notebook --no-browser --NotebookApp.allow_origin='*' --NotebookApp.token='' --ip=0.0.0.0 --NotebookApp.allow_remote_access=True"]