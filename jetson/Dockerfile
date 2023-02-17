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

# Note: The default triton docker from Deepstream have development
# libraries and headers needed to build reference applications
# Uncomment line #28 and comment out the line #30
# to use samples as base docker for a lighter container image.

# FROM nvcr.io/nvidia/deepstream-l4t:6.2-samples

FROM nvcr.io/nvidia/deepstream-l4t:6.2-triton

# ADD the triton backend installation script and execute the same
# Note: Users need to update the URL to procure newer Triton release
# at jetson/deps/misc/triton_backend_setup.sh

ADD deps/misc/triton_backend_setup.sh /opt/nvidia/deepstream/deepstream/samples/triton_backend_setup.sh
RUN /opt/nvidia/deepstream/deepstream/samples/triton_backend_setup.sh
