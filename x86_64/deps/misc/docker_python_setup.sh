#!/bin/bash
#
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
#
apt-get install software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update
apt install python3.6 -y
apt install -y libpython3.6
apt remove python3-gi -y
cd /usr/lib/python3/dist-packages/
wget http://mirrors.edge.kernel.org/ubuntu/pool/main/p/pygobject/python3-gi_3.26.1-2ubuntu1_amd64.deb
dpkg -i python3-gi_3.26.1-2ubuntu1_amd64.deb
ln -s /lib/x86_64-linux-gnu/libffi.so.7 /lib/x86_64-linux-gnu/libffi.so.6
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6m 2
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 3
#assume python3.6m is selection 2
echo 2 | update-alternatives --config python3
pip3 install numpy
pip3 install opencv-python
