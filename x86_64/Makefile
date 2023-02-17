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
img_type = triton-devel
prefix = deepstream_x86
platform = x86_64
ds_dir = /opt/nvidia/deepstream/deepstream-6.2
platform_build_arg = --build-arg TENSORRT_VERSION="8.5.2-1+cuda11.8"
platform_build_arg += --build-arg CUDNN_VERSION_2="8.7.0.84-1+cuda11.8"
platform_build_arg += --build-arg CUDA_VERSION="11.8.0"
platform_build_arg += --build-arg DS_DIR=$(ds_dir)

DOCKER=docker
triton-devel: trtserver_base_devel
triton-devel: base_image ?= $(registry)/$(prefix)-trtserver_base_devel:$(image_tag)

trtis triton-devel: target_opts += --entrypoint bash
trtis triton-devel: commit_opts += --change "ENTRYPOINT [\"$(ds_dir)/entrypoint.sh\"]"

include ../common/Makefile

ds_pkg="deepstream_sdk_v$(version)_x86_64.tbz2"
ds_pkg_dir="deepstream_sdk_v$(version)_x86_64"
platform_run_arg += -v $(top)/$(platform)/$(ds_pkg):/root/tmp/$(ds_pkg):ro
