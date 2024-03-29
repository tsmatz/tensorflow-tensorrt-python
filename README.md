# Speed up Inference by TensorRT

This example shows how to run inferencing on TensorRT.

1. [TensorRT Inferencing with ONNX Conversion](./tf_onnx_convert.ipynb)
2. [TensorFlow TensorRT Integration (TF-TRT)](./tf_trt_integration.ipynb)

Example 1 is tested with the following environment :

- Virtual Machines : Azure Standard NC4as T4 v3 (NVIDIA Tesla T4)
- Operating systems : Ubuntu 18.04
- CUDA 11.3 with cuDNN 8.2
- Python 3.6
- TensorFlow 1.15.5
- TensorRT 8.0

To run example 2, you need TensorRT 5.x.

I'll show you how to set up environment for example 1 as follows.

## How to setup and install

1. Create Ubuntu Server 18.04 LTS on Standard NC4as T4 v3 in Microsoft Azure.<br>
To run Tesla T4 instance (VM), please increase (request) quota in your Azure subscription.

> Note : You can also use [NVIDIA GPU-optimized VMI](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/nvidia.ngc_azure_17_11), in which NVIDIA drivers are already configured.

2. Python 3.6 is already installed in this virtual machine (VM).<br>
Login to this VM and check whether Python 3.6 is installed. (If not, please install Python version 3.6.)

```bash
python3 -V
```

3. Install build tools (or build-essential).

```bash
sudo apt-get update
sudo apt install -y gcc
sudo apt-get install -y make
```

4. Go to the following site, and check the supported TensorRT version and corresponding CUDA version in NVIDIA repository.<br>
In this example, I assume **TensorRT version 8.0** and **CUDA version 11.3**. (```tensorrt_8.0.1.6-1+cuda11.3_amd64.deb``` package)

[https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/)

5. Download and install CUDA.

```bash
# Install CUDA
wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run
sudo sh cuda_11.3.1_465.19.01_linux.run

# Set PATH and LD_LIBRARY_PATH for CUDA
echo 'export PATH=/usr/local/cuda-11.3/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

6. Verify whether CUDA is correctly installed. (GPU will be detected by the following command.)

```bash
nvidia-smi
```

7. Download cuDNN (runtime, dev, and samples) from NVIDIA developer site.<br>
[https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)<br>
And install the downloaded packages as follows.

```bash
sudo dpkg -i libcudnn8_8.2.1.32-1+cuda11.3_amd64.deb
sudo dpkg -i libcudnn8-dev_8.2.1.32-1+cuda11.3_amd64.deb
sudo dpkg -i libcudnn8-samples_8.2.1.32-1+cuda11.3_amd64.deb
```

8. Update PIP.

```bash
sudo apt-get update
sudo apt-get -y install python3-pip
sudo -H pip3 install --upgrade pip
```

9. For preparation of TensorRT installation, add NVIDIA package repository.

```bash
# Install key
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
# Add NVIDIA repository
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
```

10. See [installation guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) and install TensorRT.<br>
Here I have downloaded TensorRT 8.0 local repo file (```nv-tensorrt-repo-ubuntu1804-cuda11.3-trt8.0.1.6-ga-20210626_1-1_amd64.deb```) and installed as follows.<br>
I note that the following command will install the latest version of TensorRT in NVIDIA repository.

```bash
# Setup for installation
os="ubuntu1804"
tag="cuda11.3-trt8.0.1.6-ga-20210626"
sudo dpkg -i nv-tensorrt-repo-${os}-${tag}_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-${os}-${tag}/*.pub
sudo apt-get update

# Install TensorRT (latest version) and dependencies
sudo apt-get install tensorrt
# In this example, we need the following module (latest version) as well
sudo apt-get install python3-libnvinfer-dev
```

> Note : By installing ```python3-libnvinfer-dev```, TensorRT python package (including ```tensorrt```) will also be installed in your Python3 environment. (When you use conda environments, please manually install pip wheel in each environments.)

When you install old version of TensorRT instead, install specific version and all dependencies as follows.

```bash
# Install TensorRT and dependencies
sudo apt-get install libnvinfer8=8.0.1-1+cuad11.3 \
  libnvinfer-plugin8=8.0.1-1+cuad11.3 \
  libnvparsers8=8.0.1-1+cuad11.3 \
  libnvonnxparsers8=8.0.1-1+cuad11.3 \
  libnvinfer-bin=8.0.1-1+cuad11.3 \
  libnvinfer-dev=8.0.1-1+cuad11.3 \
  libnvinfer-plugin-dev=8.0.1-1+cuad11.3 \
  libnvparsers-dev=8.0.1-1+cuad11.3 \
  libnvonnxparsers-dev=8.0.1-1+cuad11.3 \
  libnvinfer-samples=8.0.1-1+cuad11.3 \
  tensorrt=8.0.1.6-1+cuda11.3
# Install python3-libnvinfer-dev and dependencies
sudo apt-get install python3-libnvinfer=8.0.1-1+cuda11.3 \
  libnvinfer8=8.0.1-1+cuda11.3 \
  libnvinfer-dev=8.0.1-1+cuda11.3 \
  libnvinfer-plugin8=8.0.1-1+cuda11.3 \
  libnvinfer-plugin-dev=8.0.1-1+cuda11.3 \
  libnvparsers8=8.0.1-1+cuda11.3 \
  libnvparsers-dev=8.0.1-1+cuda11.3 \
  libnvonnxparsers8=8.0.1-1+cuda11.3 \
  libnvonnxparsers-dev=8.0.1-1+cuda11.3 \
  python3-libnvinfer-dev=8.0.1-1+cuda11.3
```

> Note : The following command will show all dependecies for TensorRT installation.<br>
> ```sudo apt-get install tensorrt=8.0.1.6-1+cuda11.3```<br>
> ```sudo apt-get install python3-libnvinfer-dev=8.0.1-1+cuda11.3```<br>
> When you don't setup cuDNN, please add the following packages previously.<br>
> ```sudo apt-get install libcudnn8=8.2.1.32-1+cuda11.3 libcudnn8-dev=8.2.1.32-1+cuda11.3```

11. Verify the TensorRT installation as follows. (Especially, check if the correct version of packages are installed.)

```bash
dpkg -l | grep TensorRT
```

12. Install the required Python packages in this example.

```bash
pip3 install numpy tensorflow-gpu==1.15.5 tf2onnx==1.8.2 pycuda protobuf==3.16.0 onnx matplotlib
```

13. Install Jupyter.

```bash
pip3 install jupyter
```

14. Download samples.

```bash
sudo apt-get install -y git
git clone https://github.com/tsmatz/tensorflow-tensorrt-python
```

## Download pre-built ResNet50 model from NVIDIA

Download and extract TensorRT samples. Copy pre-built ResNet-50 classification graph (resnetV150_frozen.pb).

```bash
wget https://developer.download.nvidia.com/devblogs/tftrt_sample.tar.xz
tar xvf tftrt_sample.tar.xz
cp tftrt/resnetV150_frozen.pb .
```

## Run

Open [tf_onnx_convert.ipynb](./tf_onnx_convert.ipynb) with Jupyter notebook and run !

See [my blog post](https://tsmatz.wordpress.com/2018/07/07/tensorrt-tensorflow-python-on-azure-tutorial/) for details.

