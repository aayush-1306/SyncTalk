FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt update && apt install -y wget curl ca-certificates sudo build-essential ffmpeg libsm6 libxext6 portaudio19-dev cmake g++ git unzip libasound2-dev libportaudio2 libportaudiocpp0 git-lfs && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y python3.10 python3.10-dev python3-pip

RUN pip3 install tensorflow==2.12.0

RUN pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

RUN wget https://raw.githubusercontent.com/ZiqiaoPeng/SyncTalk/refs/heads/main/requirements.txt

RUN pip3 install -r requirements.txt

RUN rm requirements.txt

RUN wget https://raw.githubusercontent.com/ZiqiaoPeng/SyncTalk/refs/heads/main/scripts/install_pytorch3d.py

RUN python3 install_pytorch3d.py

RUN rm install_pytorch3d.py

RUN pip3 install ffmpeg-python

# Caching pytorch models for faster loading

RUN curl https://download.pytorch.org/models/resnet18-5c106cde.pth --create-dirs -o /root/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth

RUN curl https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth -o /root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth

RUN curl https://download.pytorch.org/models/alexnet-owt-7be5be79.pth -o /root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth

RUN curl https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip -o /root/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip

RUN curl https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth -o /root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth
