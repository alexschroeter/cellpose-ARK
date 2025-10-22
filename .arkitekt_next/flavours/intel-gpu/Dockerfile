FROM intel/intel-optimized-pytorch:2.3.110-xpu-pip-base

RUN apt update && apt install -y python3-pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN pip install arkitekt-next[all]

RUN mkdir /app
WORKDIR /app
COPY .arkitekt_next /app/.arkitekt_next
COPY app.py /app/app.py

COPY cellpose /app/cellpose
# Set a development version for setuptools-scm since we're installing local changes
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CELLPOSE=0.0.0.dev0
RUN pip install /app/cellpose/

RUN pip install matplotlib segment_anything kraph transformers