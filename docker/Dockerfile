# @brief  Docker image to install the environment to run the code in this repository.
# @author Lucas Fidon (lucas.fidon@gmail.com).
# @date   March 2022.

# Image info
FROM nvcr.io/nvidia/pytorch:21.08-py3
LABEL maintainer="lucas.fidon@gmail.com"
LABEL version="0.1"
LABEL description="Docker for trustworthy AI."

# Add third-party softwares
ADD docker/third-party third-party

# Install commonly used packages
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt install -y \
    aptitude sudo vim zsh wget curl git net-tools \
    rxvt-unicode tmux thunar locales build-essential cmake \
    python3-pip
RUN echo "alias python=python3" >> ~/.bash_aliases

# Remove unnecesary packages
RUN DEBIAN_FRONTEND=noninteractive apt remove -y --purge gdm3 xserver-xephyr

# Install locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Create folders and download pre-trained backbone AI model
RUN cd third-party \
    && mkdir nnUNet_trained_models \
    && mkdir nnUNet_raw_data_base \
    && mkdir nnUNet_preprocessed
RUN cd third-party/nnUNet_trained_models && mkdir nnUNet
RUN cd third-party/nnUNet_trained_models/nnUNet && mkdir 3d_fullres
RUN pip3 install gdown
RUN cd third-party/nnUNet_trained_models/nnUNet/3d_fullres \
#    && gdown https://drive.google.com/uc\?export\=download\&id\=1DC5qVOrarjkt8NhSIUFazKgqgLVcuWco
    && gdown https://drive.google.com/uc\?export\=download\&id\=19YjsPp_1g3FyXsthg2E47ckg1fu11c6l
#RUN cd third-party/nnUNet_trained_models/nnUNet/3d_fullres && unzip Task225_FetalBrain3dTrust.zip
RUN cd third-party/nnUNet_trained_models/nnUNet/3d_fullres && unzip Task240_FetalBrain3d_554.zip

# nnU-Net installation (used for the backbone AI)
ENV nnUNet_raw_data_base="/workspace/third-party/nnUNet_raw_data_base"
ENV nnUNet_preprocessed="/workspace/third-party/nnUNet_preprocessed"
ENV RESULTS_FOLDER="/workspace/third-party/nnUNet_trained_models"
ENV MKL_THREADING_LAYER="GNU"
RUN pip3 install git+https://github.com/LucasFidon/HardnessWeightedSampler.git
RUN cd third-party/nnUNet && pip3 install -e .

# NiftyReg (used for the fallback)
ARG NIFTYREG="/workspace/third-party/niftyreg/build/reg-apps"
ENV PATH="$PATH:$NIFTYREG"

# Install python libraries
RUN pip3 install pyyaml nested_dict==1.61 seaborn==0.11.2 loguru

WORKDIR /workspace/trustworthy-ai-fetal-brain-segmentation/

CMD "bash"
