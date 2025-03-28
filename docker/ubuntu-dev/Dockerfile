# syntax=docker/dockerfile:1

# Development dockerfile

# Load base image
FROM ubuntu:noble

ENV DEBIAN_FRONTEND=noninteractive

# Get updates
RUN apt-get update -y

#### Build and test packages
RUN apt-get install -y zsh \
  build-essential \
  cmake \
  vim \
  git \
  ninja-build \
  mold \
  ccache \
  clang \
  gdb \
  valgrind \
  catch2 \
  libbenchmark-dev \
  libeigen3-dev \
  doxygen \
  python3-pip

#### Install Documentation packages
ARG REQUIREMENTS_PATH
COPY $REQUIREMENTS_PATH requirements.txt
RUN pip install -r requirements.txt --break-system-packages
RUN rm requirements.txt

WORKDIR /usr/src

# run this
CMD ["zsh"]
