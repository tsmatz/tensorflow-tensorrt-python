# Speed up Inference by TensorRT on Azure

Tested with the following platform :

- Virtual Machines : Azure NC series (NVIDIA K80), NC v3 series (NVIDIA V100 - Volta architecture)
- Operating systems : Ubuntu 16.04
- CUDA 9.0 with cuDNN 7.1.3
- Python 3.5.2
- TensorFlow 1.8
- TensorRT 4.0.1.6

## How to setup and install

Azure Data Science Virtual Machine (DSVM) includes TensorRT (see [here](https://docs.microsoft.com/fi-FI/azure/machine-learning/data-science-virtual-machine/dsvm-deep-learning-ai-frameworks#tensorrt)), but here I use TensorFlow-TensorRT integration with python (which needs TensorFlow 1.7 or later). For this reason, here we use plain Ubuntu LTS and install drivers by ourselves.

1. Create Ubuntu 16.06 LTS on NC or NCv3 instance in Azure.

2. Login

3. Update and upgrade your system

```bash
sudo apt-get update
sudo apt-get upgrade
# python 2 / 3 is already installed ...
sudo apt install python3-pip
```

4. Install linux kernel header

```bash
sudo apt-get install linux-headers-$(uname -r)
```

5. Install CUDA 9.0 Toolkit

```bash
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda-9.0
```

6. Restart

```bash
sudo reboot
```

7. Change .bashrc

Add following lines in .bashrc.

```bash
echo 'export PATH=/usr/local/cuda-9.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

Read new settings and verify whether if you can use CUDA toolkit. (GPU info is displayed.)

```bash
source ~/.bashrc
sudo ldconfig
nvidia-smi
```

8. Install and set dependencies

```bash
sudo apt-get install libcupti-dev
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig
```

9. Install cuDNN 7.1

First you must download the following 3 files from NVIDIA.

- libcudnn7_7.1.4.18-1+cuda9.0_amd64.deb
- libcudnn7-dev_7.1.4.18-1+cuda9.0_amd64.deb
- libcudnn7-doc_7.1.4.18-1+cuda9.0_amd64.deb

Install these files.

```bash
sudo dpkg -i libcudnn7_7.1.4.18-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.1.4.18-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-doc_7.1.4.18-1+cuda9.0_amd64.deb
```

Verify whether if cuDNN is correctly installed. (It shows "Test passed!" if succeeded.)

```bash
cp -r /usr/src/cudnn_samples_v7/ $HOME
cd $HOME/cudnn_samples_v7/mnistCUDNN
make clean && make
./mnistCUDNN
cd
```

10. Restart

```bash
sudo reboot
```

11. Setup TensorRT for TensorFlow-TensorRT integration

```bash
wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvinfer-runtime-trt-repo-ubuntu1604-4.0.1-ga-cuda9.0_1-1_amd64.deb
sudo dpkg -i nvinfer-runtime-trt-repo-ubuntu1604-4.0.1-ga-cuda9.0_1-1_amd64.deb
sudo apt-get update
sudo apt-get install -y --allow-downgrades libnvinfer-dev libcudnn7-dev=7.1.4.18-1+cuda9.0 libcudnn7=7.1.4.18-1+cuda9.0
```

12. Install TensorFlow 1.8

```bash
sudo apt-mark hold libcudnn7 libcudnn7-dev
wget https://files.pythonhosted.org/packages/f2/fa/01883fee1cdb4682bbd188edc26da5982c459e681543bb7f99299fca8800/tensorflow_gpu-1.8.0-cp35-cp35m-manylinux1_x86_64.whl
pip3 install tensorflow_gpu-1.8.0-cp35-cp35m-manylinux1_x86_64.whl
sudo apt-mark unhold libcudnn7 libcudnn7-dev
```

Verify TensorFlow installation.

```bash
python3
>>> import tensorflow as tf
>>> test = tf.constant('test')
>>> sess = tf.Session() # GPU is attached here !
>>> sess.run(test)
b'test'
>>> quit()
```

13. Install Jupyter notebook

```bash
sudo apt-get -y install ipython ipython-notebook
sudo pip3 install jupyter
```

## Download pre-built ResNet-50 model from NVIDIA

Download and extract TensorRT samples. Copy pre-built ResNet-50 classification graph (resnetV150_frozen.pb).

```bash
wget https://developer.download.nvidia.com/devblogs/tftrt_sample.tar.xz
tar xvf tftrt_sample.tar.xz
cp tftrt/resnetV150_frozen.pb .
```

## Run

Open tftensorrt.ipynb with Jupyter notebook and run !

See [my blog post](https://tsmatz.wordpress.com/2018/07/07/tensorrt-tensorflow-python-on-azure-tutorial/) for details.

