# A Dockerfile that sets up a full Gym install
FROM ubuntu:16.04

# Otherwise when on AWS it asks for a keyboard layout and hangs
# See https://github.com/phusion/baseimage-docker/issues/342
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install -y software-properties-common vim
RUN add-apt-repository ppa:jonathonf/python-3.6 \
    && apt-get update \
    && apt-get install -y libav-tools \
    build-essential \
    python3.6 \
    python3.6-dev \
    python3-pip \
    libpq-dev \
    libjpeg-dev \
    curl \
    cmake \
    fceux \
    swig \
    freeglut3 \
    python-opengl \
    libboost-all-dev \
    libffi-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libsdl2-2.0-0\
    libgles2-mesa-dev \
    libsdl2-dev \
    wget \
    unzip \
    git \
    xserver-xorg-input-void \
    xserver-xorg-video-dummy \
    python-gtkglext1 \
    xpra \
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN mv /usr/bin/python /usr/bin/python2 && ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /usr/local
COPY ./docker_entrypoint .
COPY ./requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /leto.ai/async-rl
ENTRYPOINT ["/usr/local/docker_entrypoint"]
