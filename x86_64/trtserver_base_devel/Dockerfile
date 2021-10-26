 # The MIT License (MIT)
 # 
 # Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 # THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 # FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 # DEALINGS IN THE SOFTWARE.

FROM nvcr.io/nvidia/tritonserver:21.08-py3

ADD nvidia-ml.list /etc/apt/sources.list.d/

ARG TENSORRT_VERSION
ARG CUDNN_VERSION
ARG DS_DIR
ARG BOOST_VERSION=1.71.0

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
#	python-libnvinfer=${TENSORRT_VERSION} \
#	python3-libnvinfer=${TENSORRT_VERSION} \
#	python-libnvinfer-dev=${TENSORRT_VERSION} \
#	python3-libnvinfer-dev=${TENSORRT_VERSION} \
	&& rm -rf /var/lib/apt/lists/* \
  ; apt autoremove

# @{ Extra Step (1) Adding uff-converter-tf and graphsurgeon-tf packages

#Note: Download into x86_64/ folder
# the said packages from the TensorRT repo link for the TRT and Ubuntu version you want to use
#Example: http://cuda-repo/release-candidates/Libraries/TensorRT/v8.0/8.0.1.6-d2333911/11.3-r465/Ubuntu18_04-x64/deb/

RUN echo "Please follow README for Required Extra Step (1) if the following installation fail for missing files"

#Uncomment the following lines of code for ADD and RUN
#to install the uff-converter-tf package

ADD uff-converter-tf_${TENSORRT_VERSION}_amd64.deb graphsurgeon-tf_${TENSORRT_VERSION}_amd64.deb /root/
RUN apt-get update && \
	DEBIAN_FRONTEND=noninteractive  apt-get install -y --no-install-recommends \
	/root/uff-converter-tf_${TENSORRT_VERSION}_amd64.deb \
	/root/graphsurgeon-tf_${TENSORRT_VERSION}_amd64.deb && \
	rm /root/uff-converter-tf_${TENSORRT_VERSION}_amd64.deb  \
	/root/graphsurgeon-tf_${TENSORRT_VERSION}_amd64.deb

# @} Adding uff-converter-tf

ADD deps/misc/libiothub_client.so /root/
ADD deps/misc/docker_python_setup.sh /root/


# @{ Extra Step (2): Installing TRT Python3 API

#Note: Download into x86_64/deps/misc folder the TensorRT- tarball installation file from:
# https://developer.nvidia.com/nvidia-tensorrt-download

RUN echo "Please follow README for Required Extra Step (2) if the following installation fail for missing files"

#Uncomment the following lines of code for ADD and RUN

ADD deps/misc/TensorRT-8.0.1.6.Linux.x86_64-gnu.cuda-11.3.cudnn8.2.tar.gz /root/
RUN pip3 install /root/TensorRT-8.0.1.6/python/tensorrt-8.0.1.6-cp38-none-linux_x86_64.whl
RUN rm -rf /root/TensorRT-8.0.1*

# @} Installing TRT Python API



COPY trtserver_base_runtime/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY trtserver_base_runtime/entrypoint.sh ${DS_DIR}/



ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video,graphics

ENV LD_LIBRARY_PATH /opt/tritonserver/lib:/usr/src/tensorrt/lib:/opt/riva/lib:$LD_LIBRARY_PATH

ENTRYPOINT ["${DS_DIR}/entrypoint.sh"]

WORKDIR ${DS_DIR}


RUN ln -s /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so && \
    ln -s /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-encode.so && \
	ln -s /usr/lib/x86_64-linux-gnu/pkgconfig/opencv4.pc /usr/lib/x86_64-linux-gnu/pkgconfig/opencv.pc

ADD deps/gRPC_installation.sh /root/
RUN /root/gRPC_installation.sh


# @{ Extra Step (3); To copy over GStreamer 1.14 library for the DS plugin nvblender.
# This step is required until DeepStream officially support Ubuntu 20.04 (GStreamer version: 1.16).
# Currently DeepStream support Ubuntu 18.04 (GStreamer version: 1.14)

# Note: Uncomment the ADD and RUN commands below when required for nvblender.

# Copying the library from GStreamer 1.14 package into the container
# ADD deps/misc/libgstbadvideo-1.0.so.0.1405.0 /usr/lib/x86_64-linux-gnu/libgstbadvideo-1.0.so.0.1405.0
# ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libgstbadvideo-1.0.so.0
# @}  To copy over GStreamer 1.14 library for the DS plugin nvblender
