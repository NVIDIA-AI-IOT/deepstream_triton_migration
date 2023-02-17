# DeepStream 6.2 Triton Migration Guide

The documentation here is intended to help customers upgrade the Triton
version from the version DeepStreamSDK was tested and released with.
This information is useful for upgrading Triton on both x86 systems with dGPU setup
and on NVIDIA Jetson devices.

Example: User may follow the instructions here to upgrade deployment of DeepStreamSDK 6.2 which
support Triton 22.09 (on dGPU) and 23.01 (on Jetson) to a more recent Triton version like 23.01 (on dGPU).

Note: Jetson Triton Migration to 23.02 will be demonstrated here once the 23.02 release is available.
Currently, DS 6.2 already use the latest Triton version, 23.01.

This repo have artefacts that can be used to generate NVIDIA DeepStream Triton Docker for x86 machines.
For Jetson, please refer to [section 1.2 Jetson Binaries Update](#12-jetson-binaries-update).

### Additional Installations to use all DeepStreamSDK Features.

With DS 6.2, DeepStream docker containers do not package libraries necessary for certain multimedia operations like audio data parsing, CPU decode, and CPU encode.
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

The current version used by DS 6.2 official release is 22.09:
``nvcr.io/nvidia/tritonserver:22.09-py3``

The base docker used in the above Dockerfile is to demonstrate migration to Triton 23.01:
``nvcr.io/nvidia/tritonserver:23.01-py3``

Note: For customers to be able to upgrade to any of the new Triton versions,
they need to confirm API & ABI compatibility with the respective Triton version.

### 1.1.1 Prerequisites; Mandatory; (DeepStreamSDK package and terminology)

1) Please download the [DeepStreamSDK release](https://developer.nvidia.com/deepstream-getting-started) x86 tarball and place it locally
in the ``$ROOT/x86_64`` folder of this repository.

2) `image_url` is the desired docker name:TAG

3) `ds_pkg` and `ds_pkg_dir` shall be the tarball file-name with and without the
tarball extension respectively.
`ds_pkg` is also the DS tarball file-path relative to `$ROOT/x86_64`
Refer to [Section 1.1.3 Build Command](#113-build-command) for sample command.

4) `base_image` is the desired container name. Please feel free to use the sample name
provided in the command above. This name is used in the triton build steps alone.
Refer to [Section 1.1.3 Build Command](#113-build-command) for sample command.

### 1.1.2 Installing Specific TensorRT Version

DeepStreamSDK 6.2 supports TensorRT 8.5.2 (TensorRT 8.5 GA Update 1).

Users can install specific TensorRT version - say TensorRT 8.5.2 by following instructions
at line 33 and uncommenting lines 38-68 on the file: `x86_64/trtserver_base_devel/Dockerfile`.

TensorRT debian should already be downloaded into `x86_64` directory from https://developer.nvidia.com/nvidia-tensorrt-8x-download
``nv-tensorrt-local-repo-ubuntu2004-8.5.2-cuda-11.8_1.0-1_amd64.deb``.

Note: Users will need to change the TensorRT version macro `${TENSORRT_VERSION}`, `${CUDNN_VERSION_2}` at `x86_64/Makefile` if installing any other version than TensorRT 8.5.2.
To obtain this version string, please first install TensorRT manually following [installation guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-852/install-guide/index.html) and check:

``dpkg -l | grep tensorrt``

``dpkg -l | grep cudnn``


### 1.1.3 Build Command

```
sudo image_url=deepstream:6.2-triton-local \
     ds_pkg=deepstream_sdk_v6.2.0_x86_64.tbz2 \
     ds_pkg_dir=deepstream_sdk_v6.2.0_x86_64/ \
     base_image=dgpu-any-custom-base-image make triton-devel -C x86_64/
```

## 1.2 Jetson binaries update

Note: Jetson Triton Migration to 23.02 will be demonstrated here once the 23.02 release is available.
Currently, DS 6.2 already use the latest Triton version, 23.01.


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

DeepStream 6.2 Triton Server API is based on Triton 22.09 (x86) and Triton 23.01 (Jetson) release.

Regarding API compatibility, if a customer wants to upgrade triton, they need to make sure:

a) new version's `tritonserver.h` is compatible with  the:

[22.09 version of tritonserver.h for x86](https://github.com/triton-inference-server/core/blob/r22.09/include/triton/core/tritonserver.h), 
[23.01 version of tritonserver.h for jetson](https://github.com/triton-inference-server/core/blob/r23.01/include/triton/core/tritonserver.h), 
and 

b) new version’s `model_config.proto` is compatible with:

[22.09 version for x86 and jetson](https://github.com/triton-inference-server/server/blob/r22.09/src/core/model_config.proto), 
[23.01 version for jetson](https://github.com/triton-inference-server/server/blob/r23.01/src/core/model_config.proto), 

To build specific Tritonserver version libs, users can follow instructions at https://github.com/triton-inference-server/server/blob/master/docs/build.md.


### 2.2 DeepStream Config file Requirement

Gst-nvinferserver plugin’s config file kept backward compatibility.
Triton model/backend’s config.pbtxt file must follow rules of 22.09’s ``model_config.proto`` for x86 and 
23.01's ``model_config.proto`` for jetson.

## 3 Ubuntu Version Requirements

### 3.1 Ubuntu 20.04 (Triton 21.02+)

DeepStream 6.2 release package inherently support Ubuntu 20.04.

Thus, the only thing to consider is API/ABI compatibility between the new Triton version and the Triton version supported by current DS release.
