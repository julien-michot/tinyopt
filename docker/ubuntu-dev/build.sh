#!/bin/bash
requirements_dir="$(dirname "$0")/../../docs/requirements.txt"
docker build --no-cache --build-arg REQUIREMENTS_PATH=$requirements_dir -f docker/ubuntu-dev/Dockerfile -m 8g -t tinyopt-ubuntu-dev:latest .