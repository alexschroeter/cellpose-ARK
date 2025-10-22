FROM rocm/pytorch:rocm6.4.2_ubuntu24.04_py3.12_pytorch_release_2.6.0

RUN apt update && apt install -y python3-pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN pip install --break-system-packages "arkitekt-next[all]>=0.17.0"

RUN mkdir /app
WORKDIR /app
COPY .arkitekt_next /app/.arkitekt_next
COPY app.py /app/app.py

COPY cellpose /app/cellpose
# Set a development version for setuptools-scm since we're installing local changes
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CELLPOSE=0.0.0.dev0
RUN pip install --break-system-packages /app/cellpose/

RUN pip install --break-system-packages matplotlib segment_anything kraph numba
