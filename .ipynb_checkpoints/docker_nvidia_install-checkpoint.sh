# Setup the stable repository and the GPG key
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Setting up NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart docker
sudo systemctl restart docker

# Testing (Optional)
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi