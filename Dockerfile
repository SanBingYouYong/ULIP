FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

RUN apt-get update && apt-get install -y \ 
    ffmpeg build-essential git libgl1-mesa-glx libsm6 libxext6 cmake-data pkg-config libhdf5-dev && \
    rm -rf /var/lib/apt/lists/*

RUN conda init && \
    conda config --set always_yes true && \
    conda config --add channels defaults

# essentiall source ~/.bashrc
SHELL ["/bin/bash", "-c"]

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements.txt file into the container's working directory.
# Ensure this file is in the same directory as your Dockerfile when building.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-binary=h5py h5py && \
    pip install --no-cache-dir --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl && \
    pip install --no-cache-dir Ninja

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir multimethod shortuuid

ENV TORCH_CUDA_ARCH_LIST="10.0 12.0"


# Copy the application code into the container's working directory.
COPY . /app

# to get pointnet2_batch_cuda  # again, copied PointNeXt from pointnext into pointnet2 due to wrong references
RUN cd models/pointnext/PointNeXt/openpoints/cpp/pointnet2_batch && \
    python setup.py install

RUN apt-get update && apt-get install nano
