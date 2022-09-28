# DeepStream 6.1.1 Triton Migration Guide

The documentation here is intended to help customers upgrade the Triton
version from the version DeepStreamSDK was tested and released with.
This information is useful for upgrading Triton on both x86 systems with dGPU setup
and on NVIDIA Jetson devices.

Example: User may follow the instructions here to upgrade deployment of DeepStreamSDK 6.1.1 which
support Triton 22.07 (on dGPU and Jetson) to a more recent Triton version like 22.08.

This repo have artefacts that can be used to generate NVIDIA DeepStream Triton Docker for x86 machines.
For Jetson, please refer to [section 1.2 Jetson Binaries Update](#12-jetson-binaries-update).

### Additional Installations to use all DeepStreamSDK Features.

With DS 6.1.1, DeepStream docker containers do not package libraries necessary for certain multimedia operations like audio data parsing, CPU decode, and CPU encode.
This change could affect processing certain video streams/files like mp4 that include audio tracks.

Please run the below script inside the docker images to install additional packages that might be necessary to use all of the DeepStreamSDK features :

``/opt/nvidia/deepstream/deepstream/user_additional_install.sh``

## 1 How to update the Triton version?

## 1.1 DeepStream x86 docker image update

DeepStream x86 Triton docker is available on [NVIDIA NGC](https://ngc.nvidia.com/catalog/containers/nvidia:deepstream).

DeepStream x86 Triton docker builds on top of Triton base docker.

Thus, if a customer wants to change the Triton version, they may do so by
updating the version number on the base docker at:
``x86_64/trtserver_base_devel/Dockerfile``

The current version is 22.07 and the base docker used in the above Dockerfile is:
``nvcr.io/nvidia/tritonserver:22.07-py3``

Note: For customers to be able to upgrade to any of the new Triton versions,
they need to confirm API & ABI compatibility with the respective Triton version.

### 1.1.1 Prerequisites; Mandatory; (DeepStreamSDK package and terminology)

1) Please download the [DeepStreamSDK release](https://developer.nvidia.com/deepstream-getting-started) x86 tarball and place it locally
in the ``$ROOT/x86_64`` folder of this repository.

2) `image_url` is the desired docker name:TAG

3) `ds_pkg` and `ds_pkg_dir` shall be the tarball file-name with and without the
tarball extension respectively. Refer to [Section 1.1.3 Build Command](#113-build-command) for sample command.

4) `base_image` is the desired container name. Please feel free to use the sample name
provided in the command above. This name is used in the triton build steps alone.
Refer to [Section 1.1.3 Build Command](#113-build-command) for sample command.

### 1.1.2 Prerequisites; Optional; (TensorRT and other third-party packages)

1) Adding uff-converter-tf and graphsurgeon-tf packages.

Note: This is an optional step to install uff-converter-tf and graphsurgeon-tf packages.

Download file link: [nv-tensorrt-repo-ubuntu2004-cuda11.6-trt8.4.1.5-ga-20220604_1-1_amd64.deb](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.4.1/local_repos/nv-tensorrt-repo-ubuntu2004-cuda11.6-trt8.4.1.5-ga-20220604_1-1_amd64.deb) from TensorRT download page.
Note: You may have to login to [developer.nvidia.com](https://developer.nvidia.com/) to download the file.  
Quick Steps:  
$ROOT is the root directory of this git repo.    
``cd $ROOT``  
``mkdir tmp``  
``dpkg-deb -R nv-tensorrt-repo-ubuntu2004-cuda11.6-trt8.4.1.5-ga-20220604_1-1_amd64.deb tmp``  
``cp tmp/var/nv-tensorrt-repo-ubuntu2004-cuda11.6-trt8.4.1.5-ga-20220604/uff-converter-tf_8.4.1-1+cuda11.6_amd64.deb x86_64/``  
``cp tmp/var/nv-tensorrt-repo-ubuntu2004-cuda11.6-trt8.4.1.5-ga-20220604/graphsurgeon-tf_8.4.1-1+cuda11.6_amd64.deb x86_64/``  

Note: Please uncomment corresponding installation commands in ``x86_64/trtserver_base_devel/Dockerfile`` to install these optional packages inside the container.

### 1.1.3 Build Command

```
sudo image_url=deepstream:6.1.1-triton-local \
     ds_pkg=deepstream_sdk_v6.1.1_x86_64.tbz2 \
     ds_pkg_dir=deepstream_sdk_v6.1.1.0_x86_64/ \
     base_image=dgpu-any-custom-base-image make triton-devel -C x86_64/
```

## 1.2 Jetson binaries update

Note: [Triton Inference Server 22.08](https://github.com/triton-inference-server/server/releases/tag/v2.25.0) does not support Jetpack.
Users will have to use [22.07](https://github.com/triton-inference-server/server/releases/download/v2.24.0) - already the latest in DS 6.1.1 triton docker image.

Triton Inference Server release packages could be found at https://github.com/triton-inference-server/server/releases.

To upgrade the triton version on Jetson DeepStream setup, we shall replace certain libraries as discussed in the Sections below.

#### Create new Jetson triton container image with new Triton version


1) Open the file ``jetson/deps/misc/triton_backend_setup.sh`` and replace the
link to jetpack tarball for the variable ``TRITON_PKG_PATH``.  
Procure the link to this file from Triton Release page at https://github.com/triton-inference-server/server/releases  
Example, to upgrade to Triton 22.07, replace the variable value with:  
 ``TRITON_PKG_PATH=https://github.com/triton-inference-server/server/releases/download/v2.24.0/tritonserver2.24.0-jetpack5.0.2.tgz``

Note: [Triton Inference Server 22.08](https://github.com/triton-inference-server/server/releases/tag/v2.25.0) does not support Jetpack.
Users will have to use [22.07](https://github.com/triton-inference-server/server/releases/download/v2.24.0) - already the latest in DS 6.1.1 triton docker image.

2) [Optional] Open file ``jetson/Dockerfile`` and use the ds6.1.1-samples container image as base
instead of ds6.1.1-triton. ds6.1.1-triton has additional development libraries and headers
to build DeepStream reference application and these may not be required for deployment.
By default, the ds6.1.1-triton container image will be used as base to create the new image.

3) Now run the following command:

```
sudo docker build jetson/ -t deepstream-l4t:6.1.1-triton-custom
```

#### Easy way to do Steps 1.2.1 and 1.2.2

1) Please Install DeepStream 6.1.1 Package on the Jetson machine.

2) Open the file: ``/opt/nvidia/deepstream/deepstream/samples/triton_backend_setup.sh``

3) Replace the link to jetpack tarball for the variable ``TRITON_PKG_PATH``.  
Procure the link to this file from Triton Release page at https://github.com/triton-inference-server/server/releases  
Example, to upgrade to Triton 22.07, replace the variable value with:  
 ``TRITON_PKG_PATH=https://github.com/triton-inference-server/server/releases/download/v2.24.0/tritonserver2.24.0-jetpack5.0.2.tgz``

Note: [Triton Inference Server 22.08](https://github.com/triton-inference-server/server/releases/tag/v2.25.0) does not support Jetpack.
Users will have to use [22.07](https://github.com/triton-inference-server/server/releases/download/v2.24.0) - already the latest in DS 6.1.1 triton docker image.

4) Now run the shell script:

a) ``cd /opt/nvidia/deepstream/deepstream/samples/``

b) ``./triton_backend_setup.sh``

### 1.2.1 Replace the backend libs for TF1 and TF2.

DS-Triton default backend location for Jetson: ``/opt/nvidia/deepstream/deepstream/lib/triton_backends``

When users upgrade triton, they need to copy/replace their backends(TF/Custom) into the above folder.

Take Triton 22.07 for example; backends could be found from https://github.com/triton-inference-server/server/releases/tag/v2.21.0.

Jetson backends and libs are in [tritonserver2.24.0-jetpack5.0.2.tgz](https://github.com/triton-inference-server/server/releases/download/v2.24.0/tritonserver2.24.0-jetpack5.0.2.tgz).
Currently, there are TF1/TF2 backends supported on Jetson.
Users can try other triton versions and backends from release packages if the new interface is API & ABI compatible with Triton 22.07.
Tensorflow backends source code are available at https://github.com/triton-inference-server/tensorflow_backend/tree/r22.07


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

For example, TritonServer 22.07 lib is in [tritonserver2.24.0-jetpack5.0.2.tgz](https://github.com/triton-inference-server/server/releases/download/v2.24.0/tritonserver2.24.0-jetpack5.0.2.tgz).
And source code is from https://github.com/triton-inference-server/server/tree/r22.07.

Note: [Triton Inference Server 22.08](https://github.com/triton-inference-server/server/releases/tag/v2.25.0) does not support Jetpack.
Users will have to use [22.07](https://github.com/triton-inference-server/server/releases/download/v2.24.0) - already the latest in DS 6.1.1 triton docker image.

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

DeepStream 6.1.1 Triton Server API is based on Triton 22.07 (x86 and Jetson) release.

Regarding API compatibility, if a customer wants to upgrade triton, they need to make sure:

a) new version's `tritonserver.h` is compatible with  the:

[22.07 version of tritonserver.h for x86 and jetson](https://github.com/triton-inference-server/core/blob/r22.07/include/triton/core/tritonserver.h), 
and 

b) new version’s `model_config.proto` is compatible with:

[22.07 version for x86 and jetson](https://github.com/triton-inference-server/server/blob/r22.07/src/core/model_config.proto), 

To build specific Tritonserver version libs, users can follow instructions at https://github.com/triton-inference-server/server/blob/master/docs/build.md.


### 2.2 DeepStream Config file Requirement

Gst-nvinferserver plugin’s config file kept backward compatibility.
Triton model/backend’s config.pbtxt file must follow rules of 22.07’s ``model_config.proto`` for x86 and 
22.07's ``model_config.proto`` for jetson.

## 3 Ubuntu Version Requirements

### 3.1 Ubuntu 20.04 (Triton 21.02+)

DeepStream 6.1.1 release package inherently support Ubuntu 20.04.

Thus, the only thing to consider is API/ABI compatibility between the new Triton version and the Triton version supported by current DS release.
