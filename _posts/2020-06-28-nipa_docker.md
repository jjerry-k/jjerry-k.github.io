---
layout: post
title: NIPA x Docker !
category: [DeepLearning]
tags: [Tools]
sitemap :
changefreq : daily
---

> 원래 NIPA GPU 서버를 대여받은 후에 포트 포워딩을 먼저 해줘야 합니다.
하지만 그 부분에 대해선 보안적인 부분이 있기 때문에 생략하겠습니다.

이번엔 NIPA 내 개인 환경 세팅에 대해 포스팅을 해보려 합니다. 

개인마다 원하는 환경이 다르기 때문에 정말 필요하죠.

물론 기본적으로 설치된 Anaconda  환경으로도 충분할 수 있지만 살짝쿵 문제가 있습니다. 

문제에 대해 살펴보겠습니다. 

NIPA GPU 서버에 접속을 하면 다음과 같은 화면이 출력됩니다.

```
Welcome to Ubuntu 16.04.6 LTS (GNU/Linux 4.4.0-179-generic x86_64)
_______________________________________________________________________________
 _   _  _____ ______   ___            ___   _____  _____ ______  _   _
| \ | ||_   _|| ___ \ / _ \          / _ \ |_   _||  __ \| ___ \| | | |
|  \| |  | |  | |_/ // /_\ \ ______ / /_\ \  | |  | |  \/| |_/ /| | | |
| . ` |  | |  |  __/ |  _  ||______||  _  |  | |  | | __ |  __/ | | | |
| |\  | _| |_ | |    | | | |        | | | | _| |_ | |_\ \| |    | |_| |
\_| \_/ \___/ \_|    \_| |_/        \_| |_/ \___/  \____/\_|     \___/

_______________________________________________________________________________

Please use one of the following commands to start the required environment with
the framework of your choice:

- for MXNet(+Keras2) with Python3 (CUDA 10.0)
        source activate mxnet_p36

- for MXNet(+Keras2) with Python2 (CUDA 10.0)
        source activate mxnet_p27

- for TensorFlow(+Keras2) with Python3 (CUDA 10.0)
        source activate tensorflow_p36

- for TensorFlow(+Keras2) with Python2 (CUDA 10.0)
        source activate tensorflow_p27

- for TensorFlow2(+Keras2) with Python3 (CUDA 10.1)
        source activate tensorflow2_p36

- for TensorFlow2(+Keras2) with Python2 (CUDA 10.1)
        source activate tensorflow2_p27

- for Theano(+Keras2) with Python3 (CUDA 9.0)
        source activate theano_p36

- for Theano(+Keras2) with Python2 (CUDA 9.0)
        source activate theano_p27

- for PyTorch with Python3 (CUDA 10.0)
        source activate pytorch_p36

- for PyTorch with Python2 (CUDA 10.0)
        source activate pytorch_p27

- for CNTK(+Keras2) with Python3 (CUDA 9.0)
        source activate cntk_p36

- for CNTK(+Keras2) with Python2 (CUDA 9.0)
        source activate cntk_p27

- for Caffe2 with Python2 (CUDA 9.0)
        source activate caffe2_p27

- for Caffe with Python2 (CUDA 8.0)
        source activate caffe_p27

- for Caffe with Python3 (CUDA 8.0)
        source activate caffe_p35

- for Chainer with Python2 (CUDA 10.0)
        source activate chainer_p27

- for Chainer with Python3 (CUDA 10.0)
        source activate chainer_p36

- for base Python2 (CUDA 10.0)
        source activate python2

- for base Python3 (CUDA 10.0)
        source activate python3

Official Conda Guide: https://docs.conda.io/projects/conda/en/latest/user-guide/
_______________________________________________________________________________

Last login: Sat Jun 27 21:06:51 2020 from 68.175.142.82
ubuntu@nipa2020-0927:~$
```

먼저 말씀드렸던 conda 환경으로 기본적으로 다양한 환경이 제공되네요. 

저의 경우 이번에 tf-nightly가 필요했습니다. 

그래서 tensorflow2, python3.6 환경을 activate  한 후 설치를 시도했죠.

```
ubuntu@nipa2020-0927:~$ source activate tensorflow2_p36
(tensorflow2_p36) ubuntu@nipa2020-0927:~$ pip install tf-nightly-gpu
Looking in indexes: http://ftp.daumkakao.com/pypi/simple
ERROR: Could not find a version that satisfies the requirement tf-nightly-gpu (from versions: none)
ERROR: No matching distribution found for tf-nightly-gpu
```

|what...?|
|:--:|
|![Untitled_2.png](https://jjerry-k.github.io/public/img/nipa_docker/Untitled_2.png)|

음....conda 버전의 문제인가 싶어서 base conda를 update  하려 했습니다.

```
ubuntu@nipa2020-0927:~$ conda update conda
Solving environment: failed

CondaUpgradeError: This environment has previously been operated on by a conda version that's newer
than the conda currently being used. A newer version of conda is required.
  target environment location: /home/ubuntu/anaconda3
  current conda version: 4.5.12
  minimum conda version: 4.8
```

|what...?|
|:--:|
|![Untitled_2.png](https://jjerry-k.github.io/public/img/nipa_docker/Untitled_2.png)|



뭐야..이건 또 왜 안되는거야... 짜증이 났습니다. 

대충 `너무 옛날 버전의 conda니까 최소 4.8로 재설치 해주세요.` 라는 내용입니다. 

`하....이건 좀 너무한데...그냥 Docker나 설치하자..` 라는 생각을 하게 되었습니다.

그럼 Docker 설치에 대해 포스팅 해보겠습니다.

Docker 에 대한 자세한 설명은 하지 않을 겁니다. 

홈페이지 혹은 Docker 에 대한 포스팅을 참고하시기 바랍니다. 

간단히 말씀드리면 OS 단계까지 가상환경을 만드는 겁니다. 

그럼 설치 방법에 대해 적겠습니다. 

```bash
# SET UP THE REPOSITORY
sudo apt-get update

sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo apt-key fingerprint 0EBFCD88

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

# INSTALL DOCKER ENGINE
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io

# VERIFY THAT DOCKER ENGINE IS INSTALLED CORRECTLY
sudo docker run hello-world
```

만약에 제대로 설치 되었다면 마지막에 다음과 같은 출력이 남습니다

```
ubuntu@nipa2020-0927:~$ sudo docker run hello-world
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
0e03bdcc26d7: Pull complete
Digest: sha256:d58e752213a51785838f9eed2b7a498ffa1cb3aa7f946dda11af39286c3db9a9
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/
```

여기까지 하시면 기본 Docker 설치는 끝났습니다. 

하지만 이것만 설치하면 GPU 는 사용하지 못합니다. 

GPU를 쓰기 위해선 [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)를 설치를 해야 합니다.

nvidia-docker는 간단히 말하면 docker 에서 데스크탑의 GPU를 사용할 수 있도록 nvidia에서 만든(?)것입니다.

설치법은 다음과 같습니다.

```bash
# Ubuntu 16.04/18.04/20.04, Debian Jessie/Stretch/Buster
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test nvidia-docker
docker run --gpus all nvidia/cuda:10.0-base nvidia-smi
```

이 또한 설치가 제대로 되었다면 마지막에 다음과 같이 `nvidia-smi` 출력이 나올 겁니다. 

```
Sun Jun 28 01:15:13 2020
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.67       Driver Version: 418.67       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  On   | 00000000:00:06.0 Off |                    0 |
| N/A   40C    P0    42W / 250W |     10MiB / 32480MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|                                                                             |
+-----------------------------------------------------------------------------+
```

|하....편안....|
|:---:|
|![Untitled_6.png](https://jjerry-k.github.io/public/img/nipa_docker/Untitled_6.png)|

이번엔 NIPA에 Docker 설치하는 과정을 포스팅 해봤습니다. 

공짜로 빌려주는 건 좋으나 환경 구축은 역시나....해야 하네요.

제가 쓰는 Docker image는 [개인적인 도커 파일](https://jjerry-k.github.io/living/2020/05/05/dockerfile/) 에 있으니 참고하세요!