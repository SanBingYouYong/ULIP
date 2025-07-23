# Use a base image with CUDA 11.3, cuDNN 8, and Ubuntu 20.04.
# The 'devel' tag includes development tools necessary for building packages.
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Set environment variables for non-interactive installation and Conda path.
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install system-level dependencies required for Miniconda and common Python packages.
# libgl1-mesa-glx, libsm6, libxext6 are often needed for graphical libraries (e.g., OpenCV).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda for Python environment management.
# -b: batch mode (no prompts)
# -p: prefix (installation path)
# conda clean: removes downloaded packages and cache to keep the image size small.
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda clean --all -f

# Create the Conda environment named 'ulip' with Python 3.7.15.
# Then, install PyTorch 1.10.1, torchvision, torchaudio, and cudatoolkit 11.3
# from the specified channels.
# 'conda run -n ulip' ensures these commands execute within the 'ulip' environment.
RUN conda create -n ulip python=3.7.15 -y && \
    conda run -n ulip conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y && \
    conda clean --all -f

RUN conda init bash

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements.txt file into the container's working directory.
# Ensure this file is in the same directory as your Dockerfile when building.
COPY requirements.txt .

# Install additional Python dependencies specified in requirements.txt using pip,
# within the 'ulip' Conda environment.
# '--no-cache-dir' helps keep the image size down.
# We use 'bash -c' here to ensure the conda environment is properly sourced
# for this specific RUN command.
RUN bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ulip && pip install --no-cache-dir -r requirements.txt"

# The SHELL instruction is still useful for subsequent RUN commands if you add more.
# However, for interactive entry, we'll handle activation in CMD.
SHELL ["/bin/bash", "-c"]

# Define the default command to run when the container starts.
# This command will first source the conda initialization script,
# then activate the 'ulip' environment, and finally start a bash shell.
# This ensures the 'ulip' environment is active when you enter the container.
CMD ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate ulip && bash"]
