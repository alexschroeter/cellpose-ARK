from ubuntu:latest

RUN apt update && \
	apt install -y \
	python3-pip \
	python3-venv \
	python3-pyqt6 \
	libxcb-cursor0

RUN python3 -m venv /opt/venv

RUN /opt/venv/bin/pip3 install cellpose[gui]

ENV DISPLAY=:0

CMD /opt/venv/bin/cellpose
