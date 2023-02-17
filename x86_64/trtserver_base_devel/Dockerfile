# SPDX-FileCopyrightText: Copyright (c) 2022-23 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

FROM nvcr.io/nvidia/tritonserver:23.01-py3

ARG TENSORRT_VERSION
ARG CUDNN_VERSION_2
ARG DS_DIR
ARG BOOST_VERSION=1.71.0


#Removing unnecessary nvidia entries in sources.list
RUN head -n -2 /etc/apt/sources.list > /tmp/tmp.txt && mv /tmp/tmp.txt /etc/apt/sources.list

#Installing specific TRT, CuDNN versions over the versions provided by the Triton base docker image
#Users may uncomment below if needed and declare the compute lib versions at x86_64/Makefile

# NOTE: The following debian should already be downloaded into x86_64 directory
# from https://developer.nvidia.com/nvidia-tensorrt-8x-download
# ADD nv-tensorrt-local-repo-ubuntu2004-8.5.2-cuda-11.8_1.0-1_amd64.deb /tmp
#RUN mkdir -p /tmp/temp_trt && \
#        cd /tmp/temp_trt && \
#        cp /tmp/nv-tensorrt-local-repo-ubuntu2004-8.5.2-cuda-11.8_1.0-1_amd64.deb  /tmp/temp_trt && \
#        ls -l /tmp/temp_trt/ && \
#        dpkg -i /tmp/temp_trt/nv-tensorrt-local-repo-ubuntu2004-8.5.2-cuda-11.8_1.0-1_amd64.deb && \
#        cp /var/nv-tensorrt-local-repo-ubuntu2004-8.5.2-cuda-11.8/nv-tensorrt-local-9A139056-keyring.gpg /usr/share/keyrings/ && \
#        apt-get update && \
#        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
#        libnvinfer8=${TENSORRT_VERSION} \
#        libnvinfer-plugin8=${TENSORRT_VERSION} \
#        libnvparsers8=${TENSORRT_VERSION} \
#        libnvonnxparsers8=${TENSORRT_VERSION} \
#        libnvinfer-bin=${TENSORRT_VERSION} \
#        libnvinfer-dev=${TENSORRT_VERSION} \
#        libnvinfer-plugin-dev=${TENSORRT_VERSION} \
#        libnvparsers-dev=${TENSORRT_VERSION} \
#        libnvonnxparsers-dev=${TENSORRT_VERSION} \
#        libnvinfer-samples=${TENSORRT_VERSION} \
#        python3-libnvinfer=${TENSORRT_VERSION} \
#        python3-libnvinfer-dev=${TENSORRT_VERSION} \
#        uff-converter-tf=${TENSORRT_VERSION} \
#        graphsurgeon-tf=${TENSORRT_VERSION} \
#        libcudnn8-dev && \
#        cd /tmp && \
#        rm -rf temp_trt && \
#        rm -rf /var/lib/apt/lists/*
#RUN apt-get update && \
#        DEBIAN_FRONTEND=noninteractive  apt-get install -y --no-install-recommends \
#        libcudnn8=${CUDNN_VERSION_2} \
#        libcudnn8-dev=${CUDNN_VERSION_2} \

# install realsense sdk for 3d camera (Optional; disabled by default)
# instructions: https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md
#RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE && \
#        apt-get update && DEBIAN_FRONTEND=noninteractive  apt-get install -y software-properties-common && \
#        add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u &&  apt-get update && \
#        DEBIAN_FRONTEND=noninteractive         apt-get install -y \
#        librealsense2-dev librealsense2-utils && \
#        rm -rf /var/lib/apt/lists/* && \
#        apt autoremove

RUN apt-get update && \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libboost-program-options${BOOST_VERSION} \
        libboost-system${BOOST_VERSION} \
        libboost-thread${BOOST_VERSION} \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-alsa \
        gstreamer1.0-x \
        libssl1.1 \
        libgstreamer1.0-0 \
        libgstrtspserver-1.0-0 \
        libjansson4 \
        librabbitmq4 \
        libuuid1 \
        libcurl4 \
        libgles2-mesa \
        libv4l-dev libv4l2rds0 \
        libgoogle-glog0v5 \
        libjson-glib-1.0-0 \
        gstreamer1.0-rtsp \
        rsyslog \
        tzdata \
        libglvnd0 \
        libgl1 \
        libglx0 \
        libegl1 \
        libgles2 \
        linux-libc-dev \
        libglew2.1 libjpeg8 libjson-glib-1.0-0 \
        gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-tools \
        gstreamer1.0-alsa \
        libjansson-dev \
        gdb bash-completion libboost-dev \
        uuid-dev libgstrtspserver-1.0-0-dbg libgstrtspserver-1.0-dev \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libglew-dev \
        libssl-dev \
        libopencv-dev \
        freeglut3-dev \
        libyaml-cpp-dev \
        libjpeg-dev \
        libcurl4-gnutls-dev \
        libjson-glib-dev \
        libboost-dev \
        librabbitmq-dev \
        libgles2-mesa-dev libgtk-3-dev libgdk3.0-cil-dev \
        pkg-config \
        libxau-dev \
        libxdmcp-dev \
        libxcb1-dev \
        libxext-dev \
        libx11-dev \
        openssl \
        linux-libc-dev \
        policykit-1 \
        git \
        gnutls-bin \
        rsyslog \
        libjsoncpp-dev \
        python3-pip \
        libksba8 \
        vim  \
        wget \
        unzip \
        python-is-python3 \
        && rm -rf /var/lib/apt/lists/* \
        ; apt autoremove

# To migrate to Triton 23.01 or later, users will have to install CUDA 11.8 alongside
# CUDA 12.0 which comes from Triton 23.01 base image
RUN echo "Please comment the following RUN command installing CUDA 11.8 if you're migrating to Triton 22.10, 22.11, or 22.12"
RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run --no-check-certificate --no-verbose && \
    chmod +x cuda_11.8.0_520.61.05_linux.run && \
    ./cuda_11.8.0_520.61.05_linux.run --help && \
    ./cuda_11.8.0_520.61.05_linux.run --silent --toolkit && rm cuda_11.8.0_520.61.05_linux.run

# @{ Extra Steps to install dependencies for DeepStreamSDK UCX plugins

ADD deps/ofed-ucx.conf /tmp/

# Install Mellanox OFED libraries: 5.7+
 RUN mkdir /opt/dev_mlnx && \
         wget -O /opt/dev_mlnx/MLNX_OFED_LINUX-5.7-1.0.2.0-ubuntu20.04-x86_64.tgz https://content.mellanox.com/ofed/MLNX_OFED-5.7-1.0.2.0/MLNX_OFED_LINUX-5.7-1.0.2.0-ubuntu20.04-x86_64.tgz --no-check-certificate --no-verbose  && \
       mv /tmp/ofed-ucx.conf /opt/dev_mlnx/ofed-ucx.conf && \
       cd  /opt/dev_mlnx && \
        tar -xf MLNX_OFED_LINUX-5.7-1.0.2.0-ubuntu20.04-x86_64.tgz && \
        cp /opt/dev_mlnx/ofed-ucx.conf /opt/dev_mlnx/MLNX_OFED_LINUX-5.7-1.0.2.0-ubuntu20.04-x86_64/ && \
        cd  /opt/dev_mlnx/MLNX_OFED_LINUX-5.7-1.0.2.0-ubuntu20.04-x86_64 && \
        ls -l /opt/dev_mlnx/MLNX_OFED_LINUX-5.7-1.0.2.0-ubuntu20.04-x86_64  && \
        chmod +x /opt/dev_mlnx/MLNX_OFED_LINUX-5.7-1.0.2.0-ubuntu20.04-x86_64/mlnxofedinstall && \
       /opt/dev_mlnx/MLNX_OFED_LINUX-5.7-1.0.2.0-ubuntu20.04-x86_64/mlnxofedinstall --config /opt/dev_mlnx/MLNX_OFED_LINUX-5.7-1.0.2.0-ubuntu20.04-x86_64/ofed-ucx.conf --without-fw-update -q --force && \
       rm -rf /opt/dev_mlnx

# Install UCX  1.13.1
RUN mkdir /opt/dev_ucx2 && \
        wget -O /opt/dev_ucx2/ucx-1.13.1.tar.gz https://github.com/openucx/ucx/releases/download/v1.13.1/ucx-1.13.1.tar.gz --no-check-certificate --no-verbose  && \
        cd  /opt/dev_ucx2 && \
        tar -xf ucx-1.13.1.tar.gz && \
        cd /opt/dev_ucx2/ucx-1.13.1 && \
        /opt/dev_ucx2/ucx-1.13.1/contrib/configure-release --prefix=/usr --enable-examples --with-java=no --with-cuda=/usr/local/cuda --enable-mt && \
        make && make install && \
        rm -rf /opt/dev_ucx2

# install the Triton grpc client library from the prebuilt release package
RUN mkdir -p  /opt/tritonclient/ && \
        wget -O /opt/tritonclient/v2.12.0_ubuntu2004.clients.tar.gz https://github.com/triton-inference-server/server/releases/download/v2.12.0/v2.12.0_ubuntu2004.clients.tar.gz --no-check-certificate --no-verbose && \
        tar xzf /opt/tritonclient/v2.12.0_ubuntu2004.clients.tar.gz -C /opt/tritonclient/ lib include && \
        rm /opt/tritonclient/v2.12.0_ubuntu2004.clients.tar.gz 

# install protobuf and half
RUN mkdir -p /opt/proto && \
    mkdir -p /tmp/temp_proto99 && \
    mkdir -p /opt/half && \
    mkdir -p /tmp/half99 && \
    cd /tmp/temp_proto99 && \
    wget https://github.com/protocolbuffers/protobuf/releases/download/v3.8.0/protoc-3.8.0-linux-x86_64.zip --no-check-certificate --no-verbose  && \
    /usr/bin/unzip /tmp/temp_proto99/protoc-3.8.0-linux-x86_64.zip -d /opt/proto && \
    cd /tmp/half99 && \
    wget https://sourceforge.net/projects/half/files/half/2.1.0/half-2.1.0.zip --no-check-certificate --no-verbose  && \
    /usr/bin/unzip /tmp/half99/half-2.1.0.zip -d /opt/half && \
    cd / && \
    rm -rf /tmp/temp_proto99 && \
    rm -rf /tmp/half99

# @} Extra Steps

COPY trtserver_base_devel/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY trtserver_base_devel/entrypoint.sh ${DS_DIR}/


ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video,graphics

ENV LD_LIBRARY_PATH /opt/tritonserver/lib:/usr/src/tensorrt/lib:/opt/riva/lib:$LD_LIBRARY_PATH

ENTRYPOINT ["${DS_DIR}/entrypoint.sh"]

WORKDIR ${DS_DIR}

RUN ln -s /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-encode.so

ADD deps/gRPC_installation.sh /root/
RUN /root/gRPC_installation.sh

RUN DEBIAN_FRONTEND=noninteractive  apt-get install -y --no-install-recommends libssl-dev

# IP requirement for removal
RUN rm -f /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstaudioparsers.so \
       /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstx264.so \
       /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstfaad.so \
       /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstvoaacenc.so 
RUN rm -f /usr/lib/x86_64-linux-gnu/libavresample* /usr/lib/x86_64-linux-gnu/libavutil* \
    /usr/lib/x86_64-linux-gnu/libavcodec* /usr/lib/x86_64-linux-gnu/libavformat* \
    /usr/lib/x86_64-linux-gnu/libavfilter*  /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstde265.so* \
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstx265.so* /usr/lib/x86_64-linux-gnu/libde265.so* /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstvpx.so*
RUN rm -f /usr/lib/x86_64-linux-gnu/libmpeg2.so* /usr/lib/x86_64-linux-gnu/libmpeg2encpp-2.1.so* /usr/lib/x86_64-linux-gnu/libmpg123.so*  \
    /usr/lib/x86_64-linux-gnu/libx265.so.179 /usr/lib/x86_64-linux-gnu/libx264.so.155 /usr/lib/x86_64-linux-gnu/libvpx.so*  \
    /usr/lib/x86_64-linux-gnu/libmpeg2convert.so*

#ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libgstbadvideo-1.0.so.0

ADD deps/user_additional_install_devel.sh /opt/
ADD deps/user_deepstream_python_apps_install.sh /opt/

RUN rm -f /root/gRPC_installation.sh /root/*deb
