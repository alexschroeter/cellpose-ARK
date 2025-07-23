FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

RUN apt update && apt install -y python3-pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN pip install --break-system-packages "arkitekt-next[all]>=0.17.0"

RUN mkdir /app
WORKDIR /app
COPY .arkitekt_next /app/.arkitekt_next
COPY app.py /app/app.py

COPY cellpose /app/cellpose

RUN pip install --break-system-packages cellpose==3.1.1.2 matplotlib torch torchvision segment_anything
