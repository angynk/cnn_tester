FROM nvcr.io/nvidia/pytorch:22.02-py3

# Specify where our MNIST data set should be downloaded to
ENV DATA_PATH="/dataset"
ENV SRC_PATH="/src"
ENV OUTPUTS_PATH="/outputs"

# Create /work and /data directories
RUN mkdir -p /work/ ${DATA_PATH}
RUN mkdir -p /work/ ${SRC_PATH}
RUN mkdir -p /work/ ${OUTPUTS_PATH}
WORKDIR /work/

# Copy the Python source code into the image
COPY ./src /work/src

# Install Dependencies
RUN apt-get update
RUN pip install -r src/requirements.txt
# Se requiere por error  libGL.so.1
RUN apt-get install libgl1 -y 

# Download dataset and install
RUN curl -L "https://app.roboflow.com/ds/HL2Q8Kl9E3?key=SrHYWAyyMg" > roboflow.zip; unzip roboflow.zip -d dataset/; rm roboflow.zip

