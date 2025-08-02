FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

RUN apt-get update && apt-get install -y \ 
    ffmpeg build-essential git libgl1-mesa-glx libsm6 libxext6 cmake-data pkg-config libhdf5-dev nano && \
    rm -rf /var/lib/apt/lists/*

RUN conda init && \
    conda config --set always_yes true && \
    conda config --add channels defaults

COPY scripts /app/scripts
COPY ulip_models /app/ulip_models
COPY utils /app/utils
COPY main.py /app/main.py

# essentiall source ~/.bashrc
SHELL ["/bin/bash", "-c"]

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements.txt file into the container's working directory.
# Ensure this file is in the same directory as your Dockerfile when building.
COPY requirements.txt .

# we install torch again at the end as some other deps tried to overwrite the version
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-binary=h5py h5py && \
    pip install --no-cache-dir --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl && \
    pip install --no-cache-dir Ninja multimethod shortuuid && \
    pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

ENV TORCH_CUDA_ARCH_LIST="10.0 12.0"

COPY models /app/models

RUN cd models/pointnext/PointNeXt/openpoints/cpp/pointnet2_batch && \
    python setup.py install

# may come new data
COPY data /app/data

# our new stuff
COPY ulip2_encoder /app/ulip2_encoder

