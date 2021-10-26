# DeepStream Triton Migration Guide

The documentation here is intended to help customers upgrade the Triton
version from the version DeepStreamSDK was tested and released with.
This information is useful for upgrading Triton on both x86 systems with dGPU setup
and on NVIDIA Jetson devices.

Example: User may follow the instructions here to upgrade deployment of DeepStreamSDK 6.0 which
support Triton 21.08 to a more recent Triton version like 21.09.

This repo have artefacts that can be used to generate NVIDIA DeepStream Triton Docker for x86 machines.
For Jetson, please refer to [section 1.2 Jetson Binaries Update](#12-jetson-binaries-update).

## 1 How to update the Triton version?

## 1.1 DeepStream x86 docker image update

DeepStream x86 Triton docker is available on [NVIDIA NGC](https://ngc.nvidia.com/catalog/containers/nvidia:deepstream).

DeepStream x86 Triton docker builds on top of Triton base docker.

Thus, if a customer wants to change the Triton version, they may do so by
updating the version number on the base docker at:
``x86_64/trtserver_base_devel/Dockerfile``

The current version is 21.08 and the base docker used in the above Dockerfile is:
``nvcr.io/nvidia/tritonserver:21.08-py3``

Note: For customers to be able to upgrade to any of the new Triton versions,
they need to confirm API & ABI compatibility with the respective Triton version.

### 1.1.1 Prerequisites; (DeepStreamSDK package and terminology)

1) Please download the [DeepStreamSDK release](https://developer.nvidia.com/deepstream-getting-started) tarball and place it locally
in the ``$ROOT/x86_64`` folder of this repository.

2) `image_url` is the desired docker name:TAG

3) `ds_pkg` and `ds_pkg_dir` shall be the tarball file-name with and without the
tarball extension respectively. Refer to [Section 1.1.3 Build Command](#113-build-command) for sample command.

4) `base_image` is the desired container name. Please feel free to use the sample name
provided in the command above. This name is used in the triton build steps alone.
Refer to [Section 1.1.3 Build Command](#113-build-command) for sample command.

### 1.1.2 Prerequisites; (TensorRT and other third-party packages)

1) Adding uff-converter-tf and graphsurgeon-tf packages.  
Download file link: [nv-tensorrt-repo-ubuntu1804-cuda11.3-trt8.0.1.6-ga-20210626_1-1_amd64.deb](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.0.1/local_repos/nv-tensorrt-repo-ubuntu1804-cuda11.3-trt8.0.1.6-ga-20210626_1-1_amd64.deb) from TensorRT download page.  
Note: You may have to login to [developer.nvidia.com](https://developer.nvidia.com/) to download the file.  
Quick Steps:  
$ROOT is the root directory of this git repo.    
``cd $ROOT``  
``mkdir tmp``  
``dpkg-deb -R nv-tensorrt-repo-ubuntu1804-cuda11.3-trt8.0.1.6-ga-20210626_1-1_amd64.deb tmp``  
``cp tmp/var/nv-tensorrt-repo-ubuntu1804-cuda11.3-trt8.0.1.6-ga-20210626/uff-converter-tf_8.0.1-1+cuda11.3_amd64.deb x86_64/``  
``cp tmp/var/nv-tensorrt-repo-ubuntu1804-cuda11.3-trt8.0.1.6-ga-20210626/graphsurgeon-tf_8.0.1-1+cuda11.3_amd64.deb x86_64/``  

2) Installing TRT Python3 API.  
Download into ``x86_64/deps/misc`` folder the TensorRT- tarball installation file - "TensorRT 8.0.1 GA for Linux x86_64 and CUDA 11.3 TAR package" from:
https://developer.nvidia.com/nvidia-tensorrt-download  
Download file link:  [TensorRT-8.0.1.6.Linux.x86_64-gnu.cuda-11.3.cudnn8.2.tar.gz](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.0.1/tars/tensorrt-8.0.1.6.linux.x86_64-gnu.cuda-11.3.cudnn8.2.tar.gz)  
3) To copy over GStreamer 1.14 library for the DS plugin nvblender.  
Note: This step is optional and is required to use ``nvblender`` plugin.  
This step is required until DeepStream officially support Ubuntu 20.04 (GStreamer version: 1.16).  
Currently DeepStream support Ubuntu 18.04 (GStreamer version: 1.14).    
a) Copy the file ``libgstbadvideo-1.0.so.0.1405.0`` from one of the DS 6.0 docker images into the host machine.
The DS docker images are available at: https://ngc.nvidia.com/catalog/containers/nvidia:deepstream  
b) Place the file ``libgstbadvideo-1.0.so.0.1405.0`` at ``x86_64/deps/misc/``  
c) Uncomment the code marked with ``Extra Step (3)`` in ``x86_64/trtserver_base_devel/Dockerfile``  

### 1.1.3 Build Command

```
sudo image_url=deepstream:6.0-triton-local \
     ds_pkg=deepstream_sdk_v6.0.0_x86_64.tbz2 \
     ds_pkg_dir=deepstream_sdk_v6.0.0_x86_64/ \
     base_image=dgpu-any-custom-base-image make triton-devel -C x86_64/
```

## 1.2 Jetson binaries update

Triton Inference Server release packages could be found at https://github.com/triton-inference-server/server/releases.

To upgrade the triton version on Jetson DeepStream setup, we shall replace certain libraries as discussed in the Sections below.

#### Easy way to do Steps 1.2.1 and 1.2.2

1) Please Install DeepStream 6.0 Package on the Jetson machine.

2) Open the file: ``/opt/nvidia/deepstream/deepstream/samples/triton_backend_setup.sh``

3) Replace the link to jetpack tarball for the variable ``TRITON_PKG_PATH``.  
Procure the link to this file from Triton Release page at https://github.com/triton-inference-server/server/releases  
Example, to upgrade to Triton 21.09, replace the variable value with:  
 ``TRITON_PKG_PATH=https://github.com/triton-inference-server/server/releases/download/v2.14.0/tritonserver2.14.0-jetpack4.6.tgz``  

4) Now run the shell script:

a) ``cd /opt/nvidia/deepstream/deepstream/samples/``

b) ``./triton_backend_setup.sh``

### 1.2.1 Replace the backend libs for TF1 and TF2.

DS-Triton default backend location for Jetson: ``/opt/nvidia/deepstream/deepstream/lib/triton_backends``

When users upgrade triton, they need to copy/replace their backends(TF/Custom) into the above folder.

Take Triton 21.08 for example; backends could be found from https://github.com/triton-inference-server/server/releases/tag/v2.13.0.

Jetson backends and libs are in [tritonserver2.13.0-jetpack4.6.tgz](https://github.com/triton-inference-server/server/releases/download/v2.13.0/tritonserver2.13.0-jetpack4.6.tgz).
Currently, there are TF1/TF2 backends supported on Jetson.
Users can try other triton versions and backends from release packages if the new interface is API & ABI compatible with Triton 21.08.
Tensorflow backends source code are available at https://github.com/triton-inference-server/tensorflow_backend/tree/r21.08


For TensorFlow-1 backends, Users should replace folder ``/opt/nvidia/deepstream/deepstream/lib/triton_backends/tensorflow1`` with specific build TensorFlow-1 backend libs. The folder tree as below

```
/opt/nvidia/deepstream/deepstream/lib/triton_backends/tensorflow1

├── libtensorflow_cc.so -> libtensorflow_cc.so.1

├── libtensorflow_cc.so.1 -> libtensorflow_triton.so.1

├── libtensorflow_framework.so -> libtensorflow_framework.so.1

├── libtensorflow_framework.so.1 -> libtensorflow_triton.so.1

├── libtensorflow_triton.so -> libtensorflow_triton.so.1

├── libtensorflow_triton.so.1

└── libtriton_tensorflow1.so
```


For TensorFlow-2 backends, users shall replace folder ``/opt/nvidia/deepstream/deepstream/lib/triton_backends/tensorflow2`` with specific build TensorFlow-2 backend libs. Folder tree as below

```
/opt/nvidia/deepstream/deepstream/lib/triton_backends/tensorflow2

├── libtensorflow_cc.so -> libtensorflow_cc.so.2

├── libtensorflow_cc.so.2 -> libtensorflow_triton.so.2

├── libtensorflow_framework.so -> libtensorflow_framework.so.2

├── libtensorflow_framework.so.2 -> libtensorflow_triton.so.2

├── libtensorflow_triton.so -> libtensorflow_triton.so.2

├── libtensorflow_triton.so.2

└── libtriton_tensorflow2.so
```

Instead of replacing the default backends location, another option is that, update gst-nvinferserver config file to user’s new backends folder.

```
       trtis { model_repo {

          backend_dir: /path/to/new/backends/

        }}
```

### 1.2.2 Replace the server lib

Users need to replace ``/opt/nvidia/deepstream/deepstream/lib/libtritonserver.so`` with a specific triton build version.
From [Triton release webpage](https://github.com/triton-inference-server/server/releases), users could find "Jetson Jetpack Support" section to download specific tritonserver libs.

For example, TritonServer 21.09 lib is in [tritonserver2.13.0-jetpack4.6.tgz](https://github.com/triton-inference-server/server/releases/download/v2.13.0/tritonserver2.13.0-jetpack4.6.tgz).
And source code is from https://github.com/triton-inference-server/server/tree/r21.08.

### 1.3 Triton Backend upgrade

Users could follow [Triton Inference Server Backend instructions](https://github.com/triton-inference-server/backend)
to build custom backend or upgrade existing model(TF/ONNX/Pytorch) backends.

To do that, make sure to checkout the same backends branch with existing Triton Server versions.
After building the backends, then copy them into DS-Triton’s backends location as below.

DS-Triton default backends locations:

X86: ``/opt/tritonserver/backends``

Jetson: ``/opt/nvidia/deepstream/deepstream/lib/triton_backends``

Users can also update gst-nvinferserver’s config file to the new backends folder.

```
       trtis { model_repo {
          backend_dir: /path/to/new/backends/
        }}
```

## 2 API and ABI Compatibility Requirements

### 2.1 Tritonserver lib upgrade

DeepStream 6.0 GA Triton Server API is based on Triton 21.08 (Jetson) and 21.08 (x86) release.

Regarding API compatibility, if a customer wants to upgrade triton, they need to make sure:

a) new version's `tritonserver.h` is compatible with  the [21.08 version of tritonserver.h](https://github.com/triton-inference-server/core/blob/r21.08/include/triton/core/tritonserver.h)
and 

b) new version’s `model_config.proto` is exactly same as [21.08 version](https://github.com/triton-inference-server/server/blob/r21.08/src/core/model_config.proto).

To build specific Tritonserver version libs, users can follow instructions at https://github.com/triton-inference-server/server/blob/master/docs/build.md.


### 2.2 DeepStream Config file Requirement

Gst-nvinferserver plugin’s config file kept backward compatibility.
Triton model/backend’s config.pbtxt file must follow rules of 21.08’s ``model_config.proto``.

## 3 Ubuntu Version Requirements

### 3.1 Ubuntu 20.04 (Triton 21.02+)

DeepStream 6.0 Triton docker release is based on Triton 21.08
which is built on top of Ubuntu 20.04.
DeepStream 6.0 release package inherently supports Ubuntu 18.04, and thus, we
need to follow these additional steps mentioned below.

a) DeepStream Python bindings are supported on python 3.6. That version of python should be setup on the machine / docker.

Below steps need to be followed to run DeepStream python apps in the x86 Triton Docker:

- Install all other packages which are getting installed via 'apt' before running `docker_python_setup.sh`

- run `docker_python_setup.sh`

```
  root@xxxxxxxxx:/opt/nvidia/deepstream/deepstream-6.0# ./docker_python_setup.sh
```

- Note: if at some point the user runs `apt --fix-all-broken`, then they'll need to run `docker_python_setup.sh` script again.

b) Azure iot sdk binary (``libiothub_client.so``) available with DeepStream 6.0 package(x86) has been compiled against libcurl3.

It should be recompiled against default libcurl (libcurl4) library available on the system.
This is done for the user and kept at ``x86_64/deps/misc/libiothub_client.so``

c) Symbolic link for python to python3 for successful compilation of librdkafka.

This is done for the user at ``x86_64/trtserver_base_devel/Dockerfile``.

d) Symbolic link for opencv.pc to opencv4.pc for successful compilation of DS apps.

This is done for the user at ``x86_64/trtserver_base_devel/Dockerfile``.

### 3.2 Ubuntu 18.04

DeepStream 6.0 release package inherently support Ubuntu 18.04.

Thus, the only thing to consider is API/ABI compatibility between the new Triton version and Triton 21.08.
