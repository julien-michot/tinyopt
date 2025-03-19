#!/bin/bash
docker build --no-cache -f docker/ubuntu-dev/Dockerfile -m 8g -t tinyopt-ubuntu-dev:latest .