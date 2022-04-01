# A Dempster-Shafer approach to trustworthy AI with application to fetal brain MRI segmentation


## System Requirements
#### Hardware requirements
To run the automatic segmentation algorithms a NVIDIA GPU with at least 8GB of memory is required.

The code has been tested with the configuration:
* 12 Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
* 1 NVIDIA GPU GeForce GTX 1070 with 8GB of memory

#### OS Requirements
The code is supported on every OS using docker.
However, it has been tested only for
* Linux Ubuntu 18.04.6 LTS
* Linux Ubuntu 20.04.3 LTS

## Installation Guide
The installation is performed using docker.

Install docker.

Install nvidia-docker.

Install the docker image using
```bash
sh build_docker.sh
```
This step takes a few minutes.

Create a docker container for the docker image
 ```twai:latest``` that was previously built, using the command
 ```bash
nvidia-docker run --ipc=host -it -v <repository-path>:/workspace/trustworthy-ai-fetal-brain-segmentation -v <data-path>:/data --name twai twai:latest
```
where ```<repository-path>``` has to be replaced by the path of the git repository on your system
and ```<data-path>``` has to be replaced by the path of a folder containing the data to be used for segmentation.

The installation has been tested for
* Docker version 20.10.12, build e91ed57


## Demo

## Instructions to Use

#### Automatic Fetal Brain 3D MRI Segmentation

#### Figures
The figures shown in the paper can be reproduced by running
```bash
sh run_make_all_figures.sh
```
After running this command, the figures will be in the folder ```\output```.

## How to Cite
