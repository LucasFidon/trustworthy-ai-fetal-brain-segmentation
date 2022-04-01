#!/usr/bin/env bash

# Build the docker image
docker build -f docker/Dockerfile -t twai --rm .