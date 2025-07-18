FROM ros:noetic

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive

# Install system tools
RUN apt-get update && apt-get install -y \
    wget gnupg2 software-properties-common

# Add NVIDIA CUDA key and repo
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update

# Install CUDA Toolkit (e.g., 11.8)
RUN apt-get install -y cuda-toolkit-11-8

# Add CUDA to PATH and LD_LIBRARY_PATH
ENV PATH="/usr/local/cuda-11.8/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}"