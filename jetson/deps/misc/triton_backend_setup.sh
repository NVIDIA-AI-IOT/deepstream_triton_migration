#!/bin/bash
################################################################################
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
################################################################################

set -e

TRITON_DOWNLOADS=/tmp/triton_server_downloads
# Note: Update TRITON_PKG_PATH with newer Triton Release URL for Jetpack.
# Visit https://github.com/triton-inference-server/server/tags for the download link
# for newer tritonserver{triton_version}-jetpack{jp_version}.tgz
TRITON_PKG_PATH=${TRITON_PKG_PATH:=https://github.com/triton-inference-server/server/releases/download/v2.24.0/tritonserver2.24.0-jetpack5.0.2.tgz}
TRITON_BACKEND_DIR=/opt/nvidia/deepstream/deepstream/lib/triton_backends/

echo "Installing Triton prerequisites ..."
if [ $EUID -ne 0 ]; then
    echo "Must be run as root or sudo"
    exit 1
fi

apt-get update && \
    apt-get install -y --no-install-recommends libb64-dev libre2-dev libopenblas-dev

echo "Creating ${TRITON_DOWNLOADS} directory ..."
mkdir -p $TRITON_DOWNLOADS

echo "Downloading ${TRITON_PKG_PATH} to ${TRITON_DOWNLOADS} ... "
wget -O $TRITON_DOWNLOADS/jetpack.tgz $TRITON_PKG_PATH

echo "Extracting the package ....."
pushd $TRITON_DOWNLOADS
tar -xvf jetpack.tgz

echo "Creating ${TRITON_BACKEND_DIR} directory ... "
mkdir -p ${TRITON_BACKEND_DIR}

echo "Copying the backends binaries ..."
cp -r lib/libtritonserver.so /opt/nvidia/deepstream/deepstream/lib
cp -r backends/* $TRITON_BACKEND_DIR

popd

ldconfig
echo "cleaning up ${TRITON_DOWNLOADS} directory ..."
rm -rf $TRITON_DOWNLOADS
