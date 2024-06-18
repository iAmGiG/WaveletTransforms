# Use NVIDIA CUDA base image with Ubuntu 20.04
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Install necessary packages
RUN apt-get update && apt-get install -y \
    git \
    openssh-client \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create .ssh directory
RUN mkdir /root/.ssh

# Copy SSH keys and config
COPY ~/.ssh/id_ed25519 /root/.ssh/id_ed25519
COPY ~/.ssh/id_ed25519.pub /root/.ssh/id_ed25519.pub
COPY ~/.ssh/known_hosts /root/.ssh/known_hosts

# Set permissions for SSH files
RUN chmod 600 /root/.ssh/id_ed25519 && \
    chmod 644 /root/.ssh/id_ed25519.pub && \
    chmod 644 /root/.ssh/known_hosts

# Copy the Git config file
COPY ~/.gitconfig /root/.gitconfig

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Set environment variables if necessary
ENV GIT_SSH_COMMAND='ssh -i /root/.ssh/id_ed25519'

# Default command to keep the container running
CMD ["tail", "-f", "/dev/null"]
