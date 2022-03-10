#!/bin/bash
################################################################################
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#
################################################################################

set -e

TRITON_DOWNLOADS=/tmp/triton_server_downloads
# Note: Update TRITON_PKG_PATH with newer Triton Release URL for Jetpack.
# Visit https://github.com/triton-inference-server/server/tags for the download link
# for newer tritonserver{triton_version}-jetpack{jp_version}.tgz
TRITON_PKG_PATH=https://github.com/triton-inference-server/server/releases/download/v2.16.0/tritonserver2.16.0-jetpack4.6.tgz
TRITON_BACKEND_DIR=/opt/nvidia/deepstream/deepstream/lib/triton_backends/

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
