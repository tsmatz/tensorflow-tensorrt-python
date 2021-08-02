# Speed up Inference by TensorRT

This example shows how to run inferencing on TensorRT.

1. [TensorRT Inferencing with ONNX Conversion](./tf_onnx_convert.ipynb)
2. [TensorFlow TensorRT Integration (TF-TRT)](./tf_trt_integration.ipynb)

Example 1 is tested with the following environment :

- Virtual Machines : Azure Standard NC6_Promo (NVIDIA Tesla K80)
- Operating systems : Ubuntu 18.04
- CUDA 11.3 with cuDNN 8.2
- Python 3.6
- TensorFlow 1.15.5
- TensorRT 8.0

To run example 2, you need TensorRT 5.x.

I'll show you how to set up environment for example 1.

## How to setup and install

1. Create Ubuntu 18.04 LTS on Standard NC6_Promo in Microsoft Azure.

> Note : Python 3.6 is already installed in this virtual machine.

2. Install build tools (or build-essential).

```bash
sudo apt-get update
sudo apt install gcc
sudo apt-get install make
```

3. Download and install CUDA.

```bash
# Install CUDA
wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run
sudo sh cuda_11.3.1_465.19.01_linux.run

# Set PATH and LD_LIBRARY_PATH for CUDA
echo 'export PATH=/usr/local/cuda-11.3/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

4. Verify whether CUDA is correctly installed. (GPU will be detected by the following command.)

```bash
nvidia-smi
```

5. Download cuDNN (runtime, dev, and samples) from NVIDIA developer site.<br>
[https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)<br>
And install the downloaded packages as follows.

```bash
sudo dpkg -i libcudnn8_8.2.1.32-1+cuda11.3_amd64.deb
sudo dpkg -i libcudnn8-dev_8.2.1.32-1+cuda11.3_amd64.deb
sudo dpkg -i libcudnn8-samples_8.2.1.32-1+cuda11.3_amd64.deb
```

6. Update PIP.

```bash
sudo apt-get update
sudo apt-get -y install python3-pip
sudo -H pip3 install --upgrade pip
```

7. For preparation of TensorRT installation, add NVIDIA package repository.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
```

8. See [installation guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) and install TensorRT.<br>
Here I have downloaded TensorRT 8.0 local repo file (```nv-tensorrt-repo-ubuntu1804-cuda11.3-trt8.0.1.6-ga-20210626_1-1_amd64.deb```) and installed as follows.

```bash
# TensorRT 8.0.1.6 Installation
os="ubuntu1804"
tag="cuda11.3-trt8.0.1.6-ga-20210626"
sudo dpkg -i nv-tensorrt-repo-${os}-${tag}_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-${os}-${tag}/7fa2af80.pub
sudo apt-get update
sudo apt-get install tensorrt
# In this example, we need the following module as well
sudo apt-get install python3-libnvinfer-dev
```

9. Verify the TensorRT installation as follows

```bash
dpkg -l | grep TensorRT
```

10. Install the required Python packages in this example.

```bash
pip3 install numpy tensorflow-gpu==1.15.5 tf2onnx==1.8.2 pycuda protobuf==3.16.0 onnx matplotlib
```

11. Install Jupyter.

```bash
pip3 install jupyter
```

12. Download samples.

```bash
git clone https://github.com/tsmatz/tensorflow-tensorrt-python.git
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

