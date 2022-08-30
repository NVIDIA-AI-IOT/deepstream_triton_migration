# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

FROM nvcr.io/nvidia/tritonserver:22.07-py3

ARG TENSORRT_VERSION
ARG CUDNN_VERSION_2
ARG DS_DIR
ARG BOOST_VERSION=1.71.0

# Fixing: CUDA Linux Repository Key Rotation
# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb -O /root/cuda-keyring_1.0-1_all.deb
RUN apt-key del 7fa2af80 && \
        dpkg -i /root/cuda-keyring_1.0-1_all.deb && \
        rm /root/cuda-keyring_1.0-1_all.deb

#Removing unnecessary nvidia entries in sources.list
RUN head -n -2 /etc/apt/sources.list > /tmp/tmp.txt && mv /tmp/tmp.txt /etc/apt/sources.list

#Installing specific TRT, CuDNN versions over the versions provided by the Triton base docker image
#Users may uncomment below if needed and declare the compute lib versions at x86_64/Makefile
#RUN apt-get update && \
#        DEBIAN_FRONTEND=noninteractive  apt-get install -y --no-install-recommends \
#        libcudnn8=${CUDNN_VERSION_2} \
#        libcudnn8-dev=${CUDNN_VERSION_2} \
#        libnvinfer8=${TENSORRT_VERSION} \
#        libnvinfer-dev=${TENSORRT_VERSION} \
#        libnvparsers8=${TENSORRT_VERSION} \
#        libnvparsers-dev=${TENSORRT_VERSION} \
#        libnvonnxparsers8=${TENSORRT_VERSION} \
#        libnvonnxparsers-dev=${TENSORRT_VERSION} \
#        libnvinfer-plugin8=${TENSORRT_VERSION} \
#        libnvinfer-plugin-dev=${TENSORRT_VERSION} \
#        python3-libnvinfer=${TENSORRT_VERSION} \
#        python3-libnvinfer-dev=${TENSORRT_VERSION}

RUN apt-get update && \
	DEBIAN_FRONTEND=noninteractive 	apt-get install -y --no-install-recommends \
	libboost-program-options${BOOST_VERSION} \
	libboost-system${BOOST_VERSION} \
	libboost-thread${BOOST_VERSION} \
	gstreamer1.0-tools \
	gstreamer1.0-plugins-good \
	gstreamer1.0-plugins-bad \
	gstreamer1.0-plugins-ugly \
	gstreamer1.0-libav \
	gstreamer1.0-alsa \
	gstreamer1.0-x \
	libssl1.1 \
	libgstreamer1.0-0 \
	libgstrtspserver-1.0-0 	libjansson4 \
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
	gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-tools gstreamer1.0-libav \
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
	git \
	rsyslog \
	vim  \
	python-is-python3 \
	&& rm -rf /var/lib/apt/lists/* \
  ; apt autoremove

# install realsense sdk for 3d camera (Optional; disabled by default)
# instructions: https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md
#RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE && \
#        apt-get update && DEBIAN_FRONTEND=noninteractive  apt-get install -y software-properties-common && \
#        add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u &&  apt-get update && \
#        DEBIAN_FRONTEND=noninteractive         apt-get install -y \
#        librealsense2-dev librealsense2-utils && \
#        rm -rf /var/lib/apt/lists/* && \
#        apt autoremove

# @{ Extra Step (1) Adding uff-converter-tf and graphsurgeon-tf packages

#Note: Download into x86_64/ folder
# the said packages from the TensorRT repo link for the TRT and Ubuntu version you want to use
#Example: http://cuda-repo/release-candidates/Libraries/TensorRT/v8.0/8.0.1.6-d2333911/11.3-r465/Ubuntu18_04-x64/deb/

RUN echo "Please follow README for Required Extra Step (1) if the following installation fail for missing files"

#Uncomment the following lines of code for ADD and RUN
#to install the uff-converter-tf package

#ADD uff-converter-tf_${TENSORRT_VERSION}_amd64.deb graphsurgeon-tf_${TENSORRT_VERSION}_amd64.deb /root/
#RUN apt-get update && \
#	DEBIAN_FRONTEND=noninteractive  apt-get install -y --no-install-recommends \
#	/root/uff-converter-tf_${TENSORRT_VERSION}_amd64.deb \
#	/root/graphsurgeon-tf_${TENSORRT_VERSION}_amd64.deb && \
#	rm /root/uff-converter-tf_${TENSORRT_VERSION}_amd64.deb  \
#	/root/graphsurgeon-tf_${TENSORRT_VERSION}_amd64.deb

# @} Adding uff-converter-tf and graphsurgeon-tf packages

COPY trtserver_base_devel/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY trtserver_base_devel/entrypoint.sh ${DS_DIR}/



ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video,graphics

ENV LD_LIBRARY_PATH /opt/tritonserver/lib:/usr/src/tensorrt/lib:/opt/riva/lib:$LD_LIBRARY_PATH

ENTRYPOINT ["${DS_DIR}/entrypoint.sh"]

WORKDIR ${DS_DIR}


RUN ln -s /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so && \
    ln -s /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-encode.so && \
	ln -s /usr/lib/x86_64-linux-gnu/pkgconfig/opencv4.pc /usr/lib/x86_64-linux-gnu/pkgconfig/opencv.pc

ADD deps/gRPC_installation.sh /root/
RUN /root/gRPC_installation.sh
