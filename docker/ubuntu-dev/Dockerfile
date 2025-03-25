# syntax=docker/dockerfile:1

# Development dockerfile

# Load base image
FROM ubuntu:noble

ENV DEBIAN_FRONTEND noninteractive

# Get updates
RUN apt-get update -y

#### Build and test packages
RUN apt-get install -y zsh gnupg curl \
  build-essential \
  cmake \
  git \
  ninja-build \
  mold \
  ccache \
  clang \
  catch2 \
  libbenchmark-dev \
  gdb \
  vim \
  valgrind

#### 3rd parties
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
  libbenchmark-dev \
  libeigen3-dev \
  libfmt-dev \
  python3-pip

#### Install packages (for documentation)
RUN pip3 install sphinx breathe sphinx_rtd_theme sphinx_sitemap --break-system-packages

WORKDIR /usr/src

# run this
CMD ["zsh"]
