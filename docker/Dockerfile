FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
LABEL maintainer="snhryt"

ARG DEBIAN_FRONTEND="noninteractive"

# General dependencies
RUN apt-get update \
    && apt-get install -qqy software-properties-common

# Git
RUN add-apt-repository ppa:git-core/ppa \
    && apt-get update \
    && apt-get install -qqy git

# Pyenv
RUN apt-get install -qqy build-essential \
                         curl \
                         libbz2-dev \
                         libncursesw5-dev \
                         libncurses5-dev \
                         libpng-dev \
                         libreadline-dev \
                         libsqlite3-dev \
                         libssl-dev \
                         llvm \
                         make \
                         wget \
                         zlib1g-dev
RUN git clone git://github.com/yyuu/pyenv.git ~/.pyenv \
    && git clone https://github.com/yyuu/pyenv-pip-rehash.git ~/.pyenv/plugins/pyenv-pip-rehash
ENV PYENV_ROOT "/root/.pyenv"
ENV PATH "${PYENV_ROOT}/bin:${PATH}"
RUN pyenv init -

# Anaconda
ARG CONDA_VERSION="anaconda3-5.0.1"
RUN pyenv install ${CONDA_VERSION} \
    && pyenv global ${CONDA_VERSION}
ENV PATH "${PYENV_ROOT}/versions/${CONDA_VERSION}/bin:${PATH}"

# Update conda and pip
RUN conda update -y conda \
    && pip install -U pip

# Tensorflow dependencies
RUN pip install --no-cache-dir Pillow \
                               h5py \
                               ipykernel \
                               jupyter \
                               matplotlib \
                               numpy \
                               pandas \
                               scipy \
                               sklearn \
    && python -m ipykernel.kernelspec

# Tensorflow with GPU
RUN pip install --no-cache-dir \
      https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp36-cp36m-linux_x86_64.whl

# Necessary packages
RUN pip install --no-cache-dir Flask \
                               Shapely \
                               h5py \
                               keras \
                               opencv-contrib-python \
                               plumbum \
                               tqdm

# For OpenCV and EAST
RUN apt-get install -qqy python-qt4 \
                         python3-tk

# Cache clean
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Jupyter setting
COPY jupyter_notebook_config.py /root/.jupyter/

RUN mkdir -p /root/.ipython/profile_default/startup
COPY auto_import.ipy /root/.ipython/profile_default/startup/

# IPython
EXPOSE 8888
# Tensorboard
EXPOSE 6006

CMD bash
